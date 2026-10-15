"""Future explicit SHA-pinned read-only adapter, never called by synthetic tests.

Import has no project reads. Not a substitute for full inherited source audit.
Uses pure design modules only; no production entry point or fitting function.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from core_neighbors import require


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(path, expected):
    require(sha256(path) == expected, "STOP_SOURCE_SHA: " + str(path))


def load_pinned(path, name, expected):
    require(name not in sys.modules, "fresh interpreter required: " + name)
    verify(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        verify(path, expected)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_frozen(project_root, r4_protocol, *, guard):
    """Future explicit load; NO fit/torch/new neural forward/label solver.

    Must be called once in a fresh interpreter AFTER production source preflight.
    The fixed transforms are exposed, but NOT executed by this function.
    """
    require(callable(guard), "production guard required")
    guard()
    config = json.loads(Path(r4_protocol).read_text(encoding="utf-8-sig"))
    r3 = Path(project_root) / "PED_REPAIRABLE_ANCHOR_SERIES_20261011/R3_WITHIN_WINDOW_RANKING_DESIGN_20261013"
    locks = [(r3 / "core_ranking.py", config["r3_core_sha"]),
             (r3 / "frozen_inputs.py", config["r3_adapter_sha"]),
             (r3 / "protocol.json", config["r3_protocol_sha"])]
    for path, sha in locks:
        verify(path, sha)
    core3 = load_pinned(locks[0][0], "core_ranking", locks[0][1])
    adapter = load_pinned(locks[1][0], "r4_pinned_r3_inputs", locks[1][1])
    data, maps, core2, contract = adapter.load_inputs(project_root, r3 / "protocol.json", guard=guard)
    for arm in ("pointwise", "pairwise"):
        guard()
        path = r3 / "production_20261013" / arm / "OOF.npz"
        expected = config[arm + "_OOF_sha"]
        verify(path, expected)
        with np.load(path, allow_pickle=False) as saved:
            require(np.array_equal(saved["sample_id"], data["ids"]), "OOF IDs/order")
            scores, slots = saved["scores"].copy(), saved["slots"].copy()
        require(np.array_equal(core3.rank4(scores), slots), "OOF rank contract")
        data[arm + "_scores"] = scores
        verify(path, expected)
    require(np.array_equal(data["pointwise_scores"], data["old_rbf_scores"]), "R2/R3 control reproduction")
    for path, sha in locks:
        verify(path, sha)
    guard()
    return data, maps, core2, contract, adapter.fold_features
