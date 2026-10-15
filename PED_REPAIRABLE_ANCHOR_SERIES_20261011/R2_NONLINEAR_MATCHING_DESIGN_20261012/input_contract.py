"""Explicit read-only input adapter; import does not load project data.

Six entry hashes are NOT the required full inherited source/output audit.
No legacy common.py/run.py imports. No B, eval, native model, GT, or solver I/O.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from core_matching import require, rank4, check_x


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_arrays(path, keys):
    with np.load(path, allow_pickle=False) as source:
        require(set(keys).issubset(source.files), f"missing NPZ keys: {path}")
        return {key: source[key].copy() for key in keys}


def check_slots(slots, n):
    slots = np.asarray(slots)
    require(slots.shape == (n, 4) and slots.dtype.kind in "iu", "slot shape/type")
    require(np.all((slots >= 0) & (slots < 20)), "slot range")
    require(np.all(np.diff(np.sort(slots, axis=1), axis=1) > 0), "duplicate slot")


def validate_data(d):
    x = check_x(d["X"])
    n = len(x)
    require(n == 384 and d["X"].dtype == np.float32, "R1 exact384/float32 schema")
    for key in ("ids", "track", "groups", "fold", "scene"):
        require(d[key].shape == (n,), f"bad {key} shape")
    require(d["ids"].dtype.kind in "iu" and len(np.unique(d["ids"])) == n, "ID integrity")
    require(np.isin(d["groups"], [0, 1, 2]).all(), "group range")
    require([int((d["groups"] == g).sum()) for g in range(3)] == [128, 128, 128], "group quotas")
    require(np.isin(d["fold"], [0, 1, 2]).all(), "fold range")
    for track in np.unique(d["track"]):
        require(len(np.unique(d["fold"][d["track"] == track])) == 1, "track crosses folds")
    excluded = d["excluded_tracks"]
    require(len(np.unique(excluded)) == 56 and not np.intersect1d(excluded, d["track"]).size,
            "discovery tracks not excluded")
    y = d["labels"]
    require(y.shape == (n, 20) and np.isin(y, [-1, 0, 1]).all(), "labels schema")
    require([int((y == k).sum()) for k in (1, 0, -1)] == [329, 7027, 324], "R1 label census")
    require(int((y == 1).any(1).sum()) == 263, "opportunity census")
    require(np.array_equal(d["verified_safe"], y == 1), "safe label disagreement")
    require(d["envelope"].shape == y.shape and not np.any((y == 1) & ~d["envelope"]), "P outside E")
    # P2 and last-observation terms must be shared within each window.
    require(np.array_equal(x[:, :, :96], np.broadcast_to(x[:, :1, :96], (n, 20, 96))), "P2 not shared")
    require(np.array_equal(x[:, :, 120:124], np.broadcast_to(x[:, :1, 120:124], (n, 20, 4))), "past terms not shared")
    expected = [(177, 86, 40), (180, 83, 37), (169, 94, 38)]
    for fold, (nt, nv, nk) in enumerate(expected):
        train, valid = split_masks(d, fold)
        found = valid & (y == 1).any(1)
        require((int(train.sum()), int(found.sum()), len(np.unique(d["track"][found]))) == (nt, nv, nk),
                "R1 fold counts changed")
    for slots in d["reference_slots"].values():
        check_slots(slots, n)
    require(np.isfinite(d["old_witness_scores"]).all() and d["old_witness_scores"].shape == (n, 20), "witness scores")
    require(np.array_equal(rank4(d["old_witness_scores"]), d["reference_slots"]["old_witness"]), "saved witness ranking")


def split_masks(d, fold):
    require(fold in (0, 1, 2), "fold must be original0/1/2")
    valid = d["fold"] == fold
    y = d["labels"]
    train = (~valid) & (y == 1).any(1) & (y == 0).any(1)
    require(not np.intersect1d(d["track"][train], d["track"][valid]).size, "track leakage")
    return train, valid


def load_r1(r1_root, design_protocol):
    """Explicit read; does not fit or audit all2119 sources/1040 old outputs."""
    root = Path(r1_root).resolve()
    protocol = json.loads(Path(design_protocol).read_text(encoding="utf-8-sig"))
    expected = protocol["entry_sha_only_not_full_source_audit"]
    def verify():
        for relative, digest in expected.items():
            path = (root / relative).resolve()
            require(path.is_relative_to(root), "entry path escape")
            require(sha256(path) == digest, f"STOP_ENTRY_SHA: {relative}")
    verify()
    d = read_arrays(root / "sample_design.npz",
                    ("X", "ids", "track", "groups", "fold", "scene", "excluded_tracks", "envelope"))
    labels = read_arrays(root / "labels.npz", ("sample_id", "labels", "verified_safe", "envelope"))
    slots = read_arrays(root / "policy_slots.npz", ("sample_id", "witness", "envelope", "fixed_random", "bottom"))
    prior = read_arrays(root / "witness/OOF.npz", ("sample_id", "scores", "slots4"))
    for source in (labels, slots, prior):
        require(np.array_equal(d["ids"], source["sample_id"]), "STOP_ORDERED_ID_MISMATCH")
    require(np.array_equal(d["envelope"], labels["envelope"]), "envelope mismatch")
    require(np.array_equal(slots["witness"], prior["slots4"]), "witness slots mismatch")
    d.update(labels=labels["labels"], verified_safe=labels["verified_safe"],
             old_witness_scores=prior["scores"],
             reference_slots={"old_witness": slots["witness"], "envelope": slots["envelope"],
                              "fixed_random": slots["fixed_random"], "bottom": slots["bottom"]})
    validate_data(d)
    verify()
    return d
