"""Contract check for the counterfactual intervention."""

import torch

from csna_mask import focal_self_only_allowed


def main():
    allowed = torch.tensor([[[1, 1, 0], [1, 1, 1], [0, 1, 1]]], dtype=torch.bool)
    original = allowed.clone(); masked = focal_self_only_allowed(allowed)
    assert torch.equal(allowed, original), "input mask was mutated"
    assert masked[0, 0].tolist() == [True, False, False]
    assert torch.equal(masked[:, 1:], original[:, 1:]), "non-focal rows changed"
    headed = original[:, None].expand(-1, 2, -1, -1).clone()
    headed_mask = focal_self_only_allowed(headed)
    assert headed_mask[:, :, 0, 0].all() and not headed_mask[:, :, 0, 1:].any()
    assert torch.equal(headed_mask[:, :, 1:], headed[:, :, 1:])
    print("[ok] focal receive-row=self-loop; tensor shape and every non-focal edge unchanged")


if __name__ == "__main__": main()

