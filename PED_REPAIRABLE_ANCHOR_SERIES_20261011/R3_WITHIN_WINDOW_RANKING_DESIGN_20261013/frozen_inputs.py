"""Explicit read-only adapter. Import does NOT read any project experiment.

Reuses SHA-pinned pure R2 design modules, never production/common.py/run.py.
Entry checks are NOT the future full inherited-source preflight.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from core_ranking import require


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    require(digest(path) == expected, f"STOP_ENTRY_SHA: {path}")


def pure_module(path, name, expected):
    path = Path(path).resolve()
    verify(path, expected)
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).resolve() == path, "module name collision")
        raise ValueError("Use a fresh interpreter for an unambiguous pinned-module load")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass and input_contract's explicit dependency
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    verify(path, expected)
    return module


def load_inputs(project_root, protocol_path, *, guard):
    """FUTURE explicit call: returns original R1 data and frozen per-fold maps.

    No fit, no neural forward, no solver labels, no worker or output writing.
    Invoke once in a fresh interpreter, AFTER full production source preflight.
    """
    require(callable(guard), "resource guard required")
    guard()
    project = Path(project_root).resolve()
    series = project / "PED_REPAIRABLE_ANCHOR_SERIES_20261011"
    r2 = series / "R2_NONLINEAR_MATCHING_DESIGN_20261012"
    recovery = r2 / "production_20261013/report_recovery_20261013"
    r1 = series / "R1_safe_repairability_supervision_20261012/record_recovery_20261012"
    config = json.loads(Path(protocol_path).read_text(encoding="utf-8-sig"))
    locks = config["entry_sha_not_full_inherited_audit"]
    for name, sha in locks.items():
        verify(r2 / name, sha)
    core = pure_module(r2 / "core_matching.py", "core_matching", locks["core_matching.py"])
    contract = pure_module(r2 / "input_contract.py", "r3_pinned_r2_input", locks["input_contract.py"])
    data = contract.load_r1(r1, r2 / "protocol.json")
    maps, locked = [], []
    for fold in range(3):
        guard()
        base = recovery / f"rbf64/fold_{fold}"
        for file, key in (("model.npz", "rbf_model_sha"), ("run_manifest.json", "rbf_manifest_sha"),
                          ("fit_contract.npz", "rbf_contract_sha")):
            path, sha = base / file, config[key][fold]
            verify(path, sha)
            locked.append((path, sha))
        with np.load(base / "model.npz", allow_pickle=False) as saved:
            model = core.restore_model({key: saved[key].copy() for key in saved.files})
        manifest = json.loads((base / "run_manifest.json").read_text(encoding="utf-8-sig"))
        train, valid = contract.split_masks(data, fold)
        require(manifest["fold"] == fold and manifest["arm"] == "rbf64", "model fold/arm")
        require(np.array_equal(data["ids"][train], manifest["train_ids"]), "train ID/order mismatch")
        require(np.array_equal(data["ids"][valid], manifest["validation_ids"]), "validation ID/order mismatch")
        require(model.mapping.arm == "rbf64", "wrong map")
        require(np.isin(model.mapping.center_keys[:, 0], data["ids"][train]).all(), "nontrain center")
        maps.append(model.mapping)  # coefficients not used as initialization
    for arm, key in (("rbf64", "rbf_OOF_sha"), ("linear", "linear_OOF_sha")):
        path = recovery / arm / "OOF.npz"
        verify(path, config[key])
        locked.append((path, config[key]))
        with np.load(path, allow_pickle=False) as saved:
            require(np.array_equal(saved["sample_id"], data["ids"]), "OOF ordered IDs")
            require(np.array_equal(core.rank4(saved["scores"]), saved["slots"]), "OOF score/rank")
            data["reference_slots"]["old_rbf" if arm == "rbf64" else "linear"] = saved["slots"].copy()
            if arm == "rbf64":
                data["old_rbf_scores"] = saved["scores"].copy()
    for path, sha in locked:
        verify(path, sha)
    for name, sha in locks.items():
        verify(r2 / name, sha)
    guard()
    return data, maps, core, contract


def fold_features(data, maps, core, contract, fold, *, guard):
    """Fixed transform only. Both objectives MUST receive these same arrays."""
    require(fold in (0, 1, 2) and callable(guard), "fold/guard")
    train, valid = contract.split_masks(data, fold)
    z = core.transform(data["X"], maps[fold], guard=guard)
    require(z.shape == (384, 20, 200), "R2 feature dimensions")
    return z[train], z[valid], train, valid
