"""New-folder-only diagnostic utilities. No generator training or source mutation."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys, time
import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASTRA = Path('C:/Users/linke/Desktop/DRSE/ASTRA_ACCURACY_Q20')
PYTHON = 'C:/coding_toobox/envs/astra2/python.exe'
TRAIN = ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_train.npz'
DEV = ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/results/cache/train_dev/eth_dev.npz'
RAW = ASTRA/'datasets/eth_ucy/processed_data_diverse/eth_data_train.npy'
NUM = ASTRA/'datasets/eth_ucy/processed_data_diverse/eth_num_train.npy'
LABEL_NAMES = ('biwi_hotel', 'crowds_zara01', 'crowds_zara02', 'crowds_zara03', 'students001', 'students003', 'uni_examples')
FIT_N = 25912
SEED = 20261010

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def write(name, value):
    path = HERE/name
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers can briefly deny delete-sharing during an atomic replace.
    # Bounded retry only; do not change ACLs or bypass a persistent denial.
    for attempt in range(6):
        try:
            os.replace(tmp,path);break
        except PermissionError as error:
            readonly=path.exists()and bool(getattr(path.stat(),'st_file_attributes',0)&1)
            if readonly or attempt==5 or getattr(error,'winerror',None)not in (5,32):raise
            time.sleep(.02*(2**attempt))

def read(name): return json.loads((HERE/name).read_text(encoding='utf-8-sig'))
def npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}

def save_npz(name, **values):
    path = HERE/name; path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(tmp, **values); os.replace(tmp, path)

def stage(name, **info):
    value = dict(stage=name, timestamp=time.time(), **info)
    write('progress.json', value)
    print('[STAGE] '+json.dumps(value, ensure_ascii=False), flush=True)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); sys.modules[name] = result
    spec.loader.exec_module(result); return result

def source_paths():
    paths = list(HERE.glob('*.py')) + list(HERE.glob('*.ps1')) + [HERE/'protocol.json']
    paths += list((ROOT/'F16_OBSERVABLE_CUE_DIAGNOSIS_DESIGN_20261003').glob('*'))
    paths += [TRAIN, DEV, RAW, NUM,
              ROOT/'GENERATION_NATIVE_DECODER_SERIES/train_encoder.npz',
              ROOT/'GENERATION_NATIVE_DECODER_SERIES/dev_encoder.npz',
              ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/train_observations.npz',
              ROOT/'GENERATION_UPSTREAM_TEMPORAL_SERIES/dev_observations.npz',
              ROOT/'REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES/run_formal_regime_selection.py',
              ROOT/'FRONT_ENCODER_SERIES/front_encoder_models.py',
              ROOT/'FRONT_ENCODER_SERIES/train_front_variant.py',
              ROOT/'E2_F16_END_TO_END_COMPARISON_SERIES/joint_v2_adapter.py',
              ROOT/'CONTROLLED_ANCHOR_CORRECTION_SERIES/controlled_anchor_correction_models.py',
              ASTRA/'data/process_eth.py', ASTRA/'data/eth_preprocessor.py']
    paths += [ASTRA/f'datasets/eth_ucy/labels/eth/{n}_train.txt' for n in LABEL_NAMES]
    for kind in ('h3', 'v2'):
        paths += [ROOT/f'OFFICIAL_ETH_UCY_CURRENT_DECODER_SERIES/EXP_156B_FOLD_LOCAL_CURRENT_DECODER/results/eth/{kind}/f16_temporal_social_cross/seed_1/val_scene_-1/best_selected.pth']
    return paths

def lock_sources(): return {str(p): sha(p) for p in source_paths() if p.is_file()}
def verify(hashes):
    for path, expected in hashes.items():
        assert Path(path).is_file() and sha(path) == expected, ('SHA changed', path)
    return len(hashes)

def old_snapshot():
    prior = json.loads((ROOT/'GENERATION_CANDIDATE_EXCHANGE_SERIES/old_results_manifest.json').read_text())['files']
    verify(prior)
    # Inherit the complete previously locked scientific experiment history.
    # Add only relevant new scientific outputs; never recursively hash unrelated
    # external-data repositories or expand the approved data-analysis scope.
    files = dict(prior)
    additional=('GENERATION_CANDIDATE_EXCHANGE_SERIES','CANDIDATE_SELECTION_VISUAL_AUDIT_20261003',
                'TRAJECTORY_SELECTION_VISUAL_REVIEW_20261003','F16_OBSERVABLE_CUE_DIAGNOSIS_DESIGN_20261003')
    for name in additional:
        directory=ROOT/name
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock', '.tmp'):
                files[str(p)] = sha(p)
    verify(files)
    write('old_results_manifest.json', dict(files=files, inherited_manifest_n=len(prior),
          added_scientific_folders=list(additional),excluded_control_files=['.memory', 'AUTONOMOUS_GENERATION_CYCLE.json'],
          unrelated_data_repositories_not_accessed_for_analysis=True))
    return files

def load_population(split):
    cache = npz(TRAIN if split == 'train' else DEV)
    enc = npz(ROOT/f'GENERATION_NATIVE_DECODER_SERIES/{split}_encoder.npz')
    obs = npz(ROOT/f'GENERATION_UPSTREAM_TEMPORAL_SERIES/{split}_observations.npz')
    ids = cache['sample_id'].astype(np.int64)
    assert np.array_equal(ids, enc['sample_id']) and np.array_equal(ids, obs['sample_id'])
    assert len(np.unique(ids)) == len(ids)
    np.testing.assert_allclose(enc['past'].reshape(-1,16), obs['observation'][:,:16], atol=1e-6, rtol=0)
    return cache, enc, obs['observation']

def events(past, future, factor=1.):
    """Offline labels ONLY. t=1 two-step velocity includes observed p[-2]."""
    n = len(past)
    joined = np.concatenate((past[:,-2:], future), 1)
    velocity = (joined[:,2:]-joined[:,:-2])/.8
    reference = (past[:,-1]-past[:,-3])/.8
    speed = np.linalg.norm(velocity, axis=-1)
    refspeed = np.linalg.norm(reference, axis=-1)
    cross = reference[:,None,0]*velocity[:,:,1]-reference[:,None,1]*velocity[:,:,0]
    dot = (reference[:,None]*velocity).sum(-1)
    angle = np.degrees(np.arctan2(cross, dot))
    turn = (refspeed[:,None] >= .2*factor) & (speed >= .2*factor) & (np.abs(angle) >= 30*factor)
    change = np.abs(speed-refspeed[:,None]) >= np.maximum(.3*factor, .5*factor*refspeed[:,None])
    start = (refspeed[:,None] <= .1*factor) & (speed >= .3*factor)
    stop = (refspeed[:,None] >= .3*factor) & (speed <= .1*factor)
    flags = np.stack((turn, change, start, stop), -1)
    persistent = flags[:,:-1] & flags[:,1:]
    onset = np.where(persistent.any(-1), np.arange(11)[None]+1, 99).min(1)
    any_event = onset < 99
    # Last-window-only changes cannot satisfy the persistence rule.
    uncertain = ~any_event & flags[:,-1].any(-1)
    types = persistent.any(1)
    first_idx = np.minimum(onset-1, 10)
    kind = np.zeros(n, np.int64)
    for j in (1, 0, 3, 2):
        take = any_event & persistent[np.arange(n), first_idx, j]
        kind[take] = j+1
    first_turn=np.where(persistent[:,:,0],np.arange(11)[None],99).min(1)
    direction = np.where(types[:,0], np.sign(angle[np.arange(n), np.minimum(first_turn,10)]), 0).astype(np.int64)
    onset_bin = np.where(onset <= 2, 0, np.where(onset <= 6, 1, 2))
    return dict(y=any_event.astype(np.int64), uncertain=uncertain, event_types=types.astype(np.int8),
                kind=kind, onset_step=onset, onset_bin=onset_bin, direction=direction,
                refspeed=refspeed, future_speed=speed, signed_angle=angle)

def raw_features(observation):
    local = observation[:,160:304].reshape(-1,9,8,2)
    focal = local[:,0]; velocity = np.diff(focal, axis=1)/.4
    accel = np.diff(velocity, axis=1)/.4
    speed = np.linalg.norm(velocity, axis=-1)
    b1 = np.concatenate((velocity[:,-1], speed[:,-1:]), 1)
    r1 = np.concatenate((focal.reshape(-1,16), velocity.reshape(-1,14), accel.reshape(-1,12), speed),1)
    valid = np.arange(1,9)[None] < observation[:,304,None]
    relative = local[:,1:]-local[:,0,None]
    pv = (relative[:,:,-1]-relative[:,:,-3])/.8
    pos = relative[:,:,-1]
    vv = (pv*pv).sum(-1)
    tca = np.clip(-(pos*pv).sum(-1)/np.maximum(vv,1e-6),0,4.8)
    dca = np.linalg.norm(pos+tca[:,:,None]*pv,axis=-1)
    distance = np.linalg.norm(pos,axis=-1)
    closing = -(pos*pv).sum(-1)/np.maximum(distance,1e-6)
    engineered = np.stack((distance, closing, tca, dca),-1)*valid[:,:,None]
    relative = relative*valid[:,:,None,None]
    r2 = np.concatenate((r1,relative.reshape(-1,128),valid.astype(np.float32),engineered.reshape(-1,32)),1)
    assert np.isfinite(r2).all()
    return dict(B1=b1.astype(np.float32), R1=r1.astype(np.float32), R2=r2.astype(np.float32))
