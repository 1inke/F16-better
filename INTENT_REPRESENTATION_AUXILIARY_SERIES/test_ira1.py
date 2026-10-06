"""Fast synthetic contracts for the matched IRA1 intervention."""
from __future__ import annotations

import json
from pathlib import Path

import torch

import intent_aux_models as ira


HERE = Path(__file__).resolve().parent
STATS = HERE.parent / "PHYSICAL_INTENT_FLOW_SERIES" / "calibration" / "intent_stats.json"


def context_grad_norm(model: ira.IntentAuxiliaryDecoder) -> float:
    return sum(
        float(parameter.grad.norm())
        for parameter in model.context.parameters()
        if parameter.grad is not None
    )


def main() -> None:
    torch.manual_seed(20260923)
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    detached = ira.IntentAuxiliaryDecoder(16, 12, stats, False)
    joint = ira.IntentAuxiliaryDecoder(16, 12, stats, True)
    # Coupling output layers are deliberately zero-initialized.  Nudge them as
    # they would be after the first flow update so the context-gradient boundary
    # itself, rather than the identity initialization, is what this test probes.
    with torch.no_grad():
        for layer in detached.intent_flow.layers:
            layer.net[-1].weight.normal_(std=0.01)
    joint.load_state_dict(detached.state_dict(), strict=True)

    encoded = torch.randn(2, 3, 8, 16)
    past = torch.randn(2, 3, 8, 2).cumsum(dim=-2) * 0.1
    future = past[:, :, -1:, :] + torch.randn(2, 3, 12, 2).cumsum(dim=-2) * 0.1

    outputs = []
    grad_norms = []
    for model in (detached, joint):
        model.set_training_target(future)
        prediction, logits = model(encoded, past)
        outputs.append((prediction.detach(), logits.detach()))
        loss = model.auxiliary_training_loss(prediction, future, logits)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite auxiliary loss")
        loss.backward()
        grad_norms.append(context_grad_norm(model))

    prediction_error = float((outputs[0][0] - outputs[1][0]).abs().max())
    logit_error = float((outputs[0][1] - outputs[1][1]).abs().max())
    if prediction_error != 0.0 or logit_error != 0.0:
        raise RuntimeError("IRA arms do not have identical candidate inference")
    if grad_norms[0] != 0.0 or not grad_norms[1] > 0.0:
        raise RuntimeError(f"Gradient boundary failed: {grad_norms}")

    detached_count = sum(p.numel() for p in detached.parameters() if p.requires_grad)
    joint_count = sum(p.numel() for p in joint.parameters() if p.requires_grad)
    if detached_count != joint_count:
        raise RuntimeError("Matched arms have unequal trainable parameter counts")
    print(json.dumps({
        "candidate_max_abs_error": prediction_error,
        "logit_max_abs_error": logit_error,
        "detached_context_aux_grad_norm": grad_norms[0],
        "joint_context_aux_grad_norm": grad_norms[1],
        "trainable_parameters_each": detached_count,
        "contract_passed": True,
    }, indent=2))


if __name__ == "__main__":
    main()
