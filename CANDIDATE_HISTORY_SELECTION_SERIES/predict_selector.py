"""Inference using only observation/candidate inputs. Labels are never loaded."""
import argparse
from pathlib import Path
import numpy as np
import torch
from selection_core import INPUT_KEYS, Normalizer, Selector, features, history_for
from run_selection_experiments import predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    with np.load(a.input, allow_pickle=False) as f:
        data = {k: f[k] for k in (*INPUT_KEYS, 'sample_id')}
    base, past, cv = features(data)
    if base.shape[-1] != ck['base_dim']:
        raise ValueError('Input feature dimension differs from checkpoint')
    norm = Normalizer.__new__(Normalizer)
    for k, v in ck['normalizer'].items():
        setattr(norm, k, v.numpy())
    x, h = norm.apply(base, history_for(past, cv, ck['variant']))
    device = torch.device(a.device)
    net = Selector(ck['base_dim'], ck['hist_dim'], ck['variant'], ck['hidden'], ck['history_hidden']).to(device)
    net.load_state_dict(ck['weights'])
    scores = data['native_logits'] + ck['alpha'] * ck['scale'] * predict(net, x, h, device)
    idx = scores.argmax(1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, sample_id=data['sample_id'], scores=scores, selected_index=idx,
                        selected_trajectory=data['candidate_final'][np.arange(len(idx)), idx])
    print(a.output)


if __name__ == '__main__':
    main()
