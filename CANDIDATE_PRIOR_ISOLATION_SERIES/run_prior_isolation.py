"""Matched ETH pilot isolating proposal score priors and learned candidate utility."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'DIRECTIONAL_REGIME_CANDIDATE_PILOT_SERIES'
spec = importlib.util.spec_from_file_location('prior_source', SOURCE / 'directional_regime_pilot.py')
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
RESULTS = HERE / 'results'
ARMS = ('low_prior_aug28', 'equal_prior_aug28', 'utility_base20', 'utility_aug28')


class PriorHead(p.DirectionalHead):
    def __init__(self, *args, utility=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.utility = utility

    def forward(self, observation, candidate, native_logits, intent_probability):
        output, scale = super().forward(observation, candidate, native_logits, intent_probability)
        if self.utility:
            return self.last_residual, torch.zeros_like(scale)
        return output, scale


def prepare_set(cache, past, future, cfg, arm):
    candidate_set = p.build_candidate_set(cache, past, future, cfg, augmented=arm != 'utility_base20')
    if arm == 'equal_prior_aug28':
        # Match the existing base candidates' median, removing only the .5-std penalty.
        candidate_set['logits'][:, 20:] = np.median(cache['logits'], axis=1, keepdims=True)
        candidate_set['features'] = p.candidate_features(
            past, candidate_set['trajectory'], candidate_set['logits'],
            candidate_set['intent_label'], candidate_set['explicit_flag'])
    if arm.startswith('utility_'):
        # Neither the additive native prior nor its feature may influence utility arms.
        candidate_set['logits'] = np.zeros_like(candidate_set['logits'])
        candidate_set['features'] = candidate_set['features'].copy()
        candidate_set['features'][..., 30] = 0.0
        candidate_set['features'][..., 38] = 0.0  # remove synthetic/native provenance cue
    return candidate_set


def verify(cfg, device):
    if not torch.cuda.is_available() and device.type == 'cuda':
        raise RuntimeError('CUDA unavailable')
    p.seed_everything(1)
    head = PriorHead(10, 39, 32, 0.0, True, .75, .2, utility=True).to(device)
    obs = torch.randn(4, 10, device=device)
    cand = torch.randn(4, 28, 39, device=device)
    native = torch.randn(4, 28, device=device)
    prob = torch.softmax(torch.randn(4, 7, device=device), -1)
    with torch.no_grad():
        head.network[-1].weight.normal_(0, .1)
    a, _ = head(obs, cand, native, prob)
    b, _ = head(obs, cand, native * 100, prob)
    assert torch.allclose(a, b, atol=1e-6), 'Utility still depends on native prior'
    permutation = torch.randperm(28, device=device)
    c, _ = head(obs, cand[:, permutation], native[:, permutation], prob)
    assert torch.allclose(a[:, permutation], c, atol=1e-6), 'Permutation check failed'
    target = torch.softmax(torch.randn_like(a), -1)
    loss = -(target * torch.log_softmax(a, -1)).sum(-1).mean()
    loss.backward()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in head.parameters())
    train = p.load_npz(p.TRAIN_CACHE)
    past, future = p.raw_focal_trajectories(train['sample_id'], train['motion'])
    tiny = {k: v[:16] for k, v in train.items()}
    sets = {arm: prepare_set(tiny, past[:16], future[:16], cfg, arm) for arm in ARMS}
    assert np.array_equal(sets['low_prior_aug28']['trajectory'], sets['equal_prior_aug28']['trajectory'])
    assert np.array_equal(sets['low_prior_aug28']['ade'], sets['equal_prior_aug28']['ade'])
    assert np.all(sets['utility_aug28']['features'][..., [30, 38]] == 0)
    assert np.array_equal(sets['utility_base20']['features'], sets['utility_aug28']['features'][:, :20])
    fit, cal = p.internal_split(len(train['sample_id']), cfg['internal_calibration_fraction'])
    sources = [Path(__file__), SOURCE / 'directional_regime_pilot.py', HERE / 'config.json',
               p.TRAIN_CACHE, p.DEV_CACHE, p.RAW_TRAIN, SOURCE / 'results/models/intent_classifier.pth']
    report = {'status': 'PASS', 'fit_n': len(fit), 'calibration_n': len(cal),
              'test_access': False, 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
              'hashes': {str(f): p.sha256(f) for f in sources}}
    (HERE / 'preflight.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PREFLIGHT PASS: prior independence, permutation, GPU gradients, candidate identity, raw-cache alignment', flush=True)


def run(cfg, device):
    start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    p.MODELS = RESULTS / 'models'
    saved_intent = torch.load(SOURCE / 'results/models/intent_classifier.pth', map_location='cpu', weights_only=False)
    classifier = p.IntentClassifier(saved_intent['input_dim'], cfg['intent_dropout']).to(device)
    classifier.load_state_dict(saved_intent['state_dict']); classifier.eval()
    train = p.load_npz(p.TRAIN_CACHE)
    past, future = p.raw_focal_trajectories(train['sample_id'], train['motion'])
    fit, cal = p.internal_split(len(past), cfg['internal_calibration_fraction'])
    normalized = p.normalized_observation(train, saved_intent['mean'], saved_intent['std'])
    probability = p.predict_intent(classifier, normalized, device)
    observation = np.concatenate((normalized, probability), axis=1).astype(np.float32)
    heads = {}
    for arm in ARMS:
        print(f'[TRAIN] {arm} | {cfg["reranker_epochs"]} epochs', flush=True)
        utility = arm.startswith('utility_')
        p.DirectionalHead = lambda *a, _utility=utility, **kw: PriorHead(*a, utility=_utility, **kw)
        p.seed_everything(cfg['seed'])  # matched initialization and batch order across arms
        candidate_set = prepare_set(train, past, future, cfg, arm)
        heads[arm] = p.train_head(arm, not utility, train, candidate_set, observation, probability,
                                  fit, cal, cfg, device)
        print(f'[COMPLETE] {arm} selected epoch={heads[arm][1]["best_epoch"]}', flush=True)
        del candidate_set
    locks = {str(f): p.sha256(f) for f in p.MODELS.glob('*.pth')}
    (RESULTS / 'model_lock.json').write_text(json.dumps({'models': locks, 'test_access': False,
        'dev_used_for_selection': False, 'time': time.time()}, indent=2), encoding='utf-8')
    print('[EVALUATE] All four models locked; ETH development diagnostic', flush=True)
    dev = p.load_npz(p.DEV_CACHE)
    past, future = p.raw_focal_trajectories(dev['sample_id'], dev['motion'])
    normalized = p.normalized_observation(dev, saved_intent['mean'], saved_intent['std'])
    probability = p.predict_intent(classifier, normalized, device)
    observation = np.concatenate((normalized, probability), axis=1).astype(np.float32)
    base = p.build_candidate_set(dev, past, future, cfg, False)
    arrays = {'native': p.selection_arrays(base['logits'], base)}
    rows = p.metric_rows('native', arrays['native'], dev['change_group'])
    explicit = {}
    for arm in ARMS:
        candidate_set = prepare_set(dev, past, future, cfg, arm)
        model, saved = heads[arm]
        logits, scale = p.predict_head(model, observation, p.transform_candidate(candidate_set['features'], saved),
                                       candidate_set['logits'], probability, device)
        arrays[arm] = p.selection_arrays(logits, candidate_set)
        rows += p.metric_rows(arm, arrays[arm], dev['change_group'], scale)
        selected_extra = arrays[arm]['selected_index'] >= 20
        explicit[arm] = {'selected_n': int(selected_extra.sum()),
                         'oracle_n': int((arrays[arm]['oracle_index'] >= 20).sum()),
                         'selected_extra_ade': float(arrays[arm]['selected_ade'][selected_extra].mean()) if selected_extra.any() else None}
        np.savez_compressed(RESULTS / f'{arm}_per_sample.npz', sample_id=dev['sample_id'],
                            logits=logits, intent_probability=probability, **arrays[arm])
    comparisons = [('equal_prior_aug28', 'low_prior_aug28'), ('utility_aug28', 'utility_base20'),
                   ('utility_base20', 'native'), ('utility_aug28', 'native'), ('equal_prior_aug28', 'native')]
    bootstrap = []
    for i, (treatment, reference) in enumerate(comparisons):
        for metric in ('selected_ade', 'selected_fde', 'oracle_ade'):
            delta, low, high = p.paired_bootstrap_delta(arrays[treatment][metric], arrays[reference][metric],
                                                       cfg['bootstrap_repetitions'], cfg['bootstrap_seed'] + i)
            bootstrap.append({'treatment': treatment, 'reference': reference, 'metric': metric,
                              'delta': delta, 'ci_low': low, 'ci_high': high})
    p.write_csv(RESULTS / 'metrics.csv', rows)
    p.write_csv(RESULTS / 'paired_bootstrap.csv', bootstrap)
    summary = {'status': 'COMPLETE_REVIEW_REQUIRED', 'elapsed_seconds': time.time() - start,
               'explicit_candidate_diagnostics': explicit, 'metrics': rows, 'paired_bootstrap': bootstrap,
               'scope': 'ETH seed1 development diagnosis; generator-seen fit/calibration; reused frozen intent classifier; no test'}
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Candidate prior isolation pilot', '', summary['scope'], '',
             '| Arm | Selected ADE | Selected FDE | minADE |', '|---|---:|---:|---:|']
    for row in rows:
        if row['group'] == 'overall':
            lines.append(f'| {row["arm"]} | {row["selected_ade"]:.6f} | {row["selected_fde"]:.6f} | {row["minade"]:.6f} |')
    lines += ['', '## Explicit proposals', '', '```json', json.dumps(explicit, indent=2), '```', '',
              '## Paired bootstrap (conditional on one trained seed)', '']
    for row in bootstrap:
        lines.append(f'- {row["treatment"]} vs {row["reference"]}, {row["metric"]}: {row["delta"]:+.6f} [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}]')
    (RESULTS / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'COMPLETE | elapsed={summary["elapsed_seconds"]:.1f}s | {RESULTS}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('preflight', 'run'), default='preflight')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
    device = torch.device(args.device)
    try:
        if args.action == 'preflight':
            verify(cfg, device)
        else:
            audit = json.loads((HERE / 'preflight.json').read_text(encoding='utf-8'))
            for source, expected in audit['hashes'].items():
                if p.sha256(Path(source)) != expected:
                    raise RuntimeError(f'Preflight source changed: {source}')
            run(cfg, device)
    except Exception:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
