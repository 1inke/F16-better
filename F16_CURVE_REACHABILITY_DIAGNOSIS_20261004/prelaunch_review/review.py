"""Additive launch review; no fitting, model forward, or scientific-source edit."""
from pathlib import Path
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

OUT = Path(__file__).resolve().parent
HERE = OUT.parent
ROOT = HERE.parent


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    preflight = read(HERE / "preflight.json")
    assert preflight["status"] == "PASS"
    assert read(HERE / "MEMORY_PREFLIGHT.json")["status"] == "PASS_FULL_INPUT_CHUNK_MEMORY_AND_ARITHMETIC"
    assert read(HERE / "resume_check.json")["status"] == "PASS"
    assert not (HERE / "RESULT.json").exists(), "Do not relaunch completed science"
    assert not list((HERE / "chunks").glob("FIT_*.done.json"))
    assert not list((HERE / "chunks").glob("CAL_*.done.json"))
    locks = preflight["source_hashes"]
    for path, expected in locks.items():
        assert digest(path) == expected, "Registered source changed: " + path
    print("[LAUNCH_REVIEW] all registered source SHA unchanged: " + str(len(locks)), flush=True)

    spec = importlib.util.spec_from_file_location("reachability_launch_review_core", HERE / "diagnosis.py")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    memory_before = core.infra.memory()
    reserve = 6 * 1024 ** 3
    assert memory_before["pointer_bits"] == 64
    assert min(memory_before["physical_available_bytes"], memory_before["system_commit_available_bytes"]) >= reserve

    import torch
    torch.set_num_threads(2)
    models = []
    for seed in (1, 2, 3):
        path = ROOT / "GENERATION_PULSE_RESPONSE_SERIES_20261004" / "results" / "models" / "mean_std" / ("seed_" + str(seed)) / "balanced_protected" / "constrained.pth"
        expected = locks[str(path)]
        assert digest(path) == expected
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        models.append({"seed": seed, "path": str(path), "sha": expected,
                       "epoch": int(checkpoint["epoch"]), "input_dim": int(checkpoint["input_dim"]),
                       "state_dict_entries": len(checkpoint["state_dict"]),
                       "metadata_read_only": True, "new_model_forward": False})
        del checkpoint
        gc.collect()
        assert digest(path) == expected

    memory_after = core.infra.memory()
    assert memory_after["private_bytes"] < reserve
    assert min(memory_after["physical_available_bytes"], memory_after["system_commit_available_bytes"]) >= reserve
    assert digest(HERE / "diagnosis.py") == locks[str(HERE / "diagnosis.py")]
    report = {
        "status": "PASS_ADDITIVE_PRELAUNCH_REVIEW",
        "checked_at": datetime.now().astimezone().isoformat(),
        "registered_source_files_SHA_unchanged": len(locks),
        "scientific_blocks_completed_before_launch": 0,
        "planned_scientific_blocks": math.ceil(25912 / 64) + math.ceil(2879 / 64),
        "learned_replay_models_metadata": models,
        "original_preflight_metadata_omission": "The existing preflight models field was empty after model cleanup; preserve it unchanged. This supplement reads the three locked checkpoints, not a repeat of the model-forward preflight.",
        "preflight_SHA": digest(HERE / "preflight.json"),
        "protocol_SHA": digest(HERE / "protocol.json"),
        "review_source_SHA": digest(__file__),
        "memory_before": memory_before,
        "memory_after": memory_after,
        "old_full_manifest_count": len(read(HERE / "old_results_manifest.json")["files"]),
        "full_old_files_prelaunch_rehashed": False,
        "full_old_hash_check": "Independent worker must verify every registered old file before scientific computation and after final saved-output audit.",
        "model_training": False, "model_forward_this_review": False,
        "scientific_protocol_changed": False, "existing_preflight_or_results_edited": False,
        "resume": "Existing done-SHA skip and bad-source-reject PASS; not optimizer/RNG training resume.",
        "inline_adversarial_gate": "PASS_WITH_LIMITATIONS: successful GT-assisted construction witnesses feasibility, failure under the fixed local budget stays unresolved. Oracle fitting is not deployed model quality or causal root-cause evidence. No new outcomes reviewed to alter the protocol."
    }
    destination = OUT / "REVIEW.json"
    assert not destination.exists(), "Preserve prior launch-review evidence"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "source_files": len(locks),
                      "models": len(models), "scientific_blocks": report["planned_scientific_blocks"],
                      "physical_available_GiB": memory_after["physical_available_bytes"] / 2 ** 30,
                      "commit_available_GiB": memory_after["system_commit_available_bytes"] / 2 ** 30,
                      "private_GiB": memory_after["private_bytes"] / 2 ** 30}), flush=True)


if __name__ == "__main__":
    main()
