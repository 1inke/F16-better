"""Deterministic batched K-medoids for the 40-to-20 representability audit."""
from __future__ import annotations

import torch
from torch import Tensor


def _selected_distances(distance: Tensor, selected: Tensor) -> Tensor:
    return distance.gather(2, selected[:, None, :].expand(-1, distance.shape[1], -1))


def deterministic_kmedoids(distance: Tensor, k: int = 20, max_swaps: int = 40) -> Tensor:
    """Central-medoid + farthest initialization followed by PAM-style swaps.

    Tie handling is deterministic because torch.argmin/argmax returns the first
    index.  The objective is the unweighted sum of teacher-to-medoid distances.
    """
    if distance.ndim != 3 or distance.shape[1] != distance.shape[2]:
        raise ValueError(f"Expected [B,N,N] distance, got {tuple(distance.shape)}")
    batch, n, _ = distance.shape
    if not 0 < k <= n:
        raise ValueError(f"Invalid k={k} for n={n}")

    selected = torch.empty((batch, k), dtype=torch.long, device=distance.device)
    mask = torch.zeros((batch, n), dtype=torch.bool, device=distance.device)
    first = distance.mean(dim=-1).argmin(dim=-1)
    selected[:, 0] = first
    mask.scatter_(1, first[:, None], True)
    nearest = distance.gather(2, first[:, None, None].expand(-1, n, 1)).squeeze(-1)
    for slot in range(1, k):
        candidate_score = nearest.masked_fill(mask, float("-inf"))
        nxt = candidate_score.argmax(dim=-1)
        selected[:, slot] = nxt
        mask.scatter_(1, nxt[:, None], True)
        new_distance = distance.gather(2, nxt[:, None, None].expand(-1, n, 1)).squeeze(-1)
        nearest = torch.minimum(nearest, new_distance)

    all_indices = torch.arange(n, device=distance.device)[None].expand(batch, -1)
    for _ in range(max_swaps):
        mask.zero_().scatter_(1, selected, True)
        unselected = all_indices.masked_fill(mask, n).sort(dim=1).values[:, : n - k]
        distances_to_selected = _selected_distances(distance, selected)
        two, order = distances_to_selected.topk(2, dim=-1, largest=False, sorted=True)
        nearest, second = two[..., 0], two[..., 1]
        nearest_slot = order[..., 0]

        removal_slot = torch.arange(k, device=distance.device)[None, :, None]
        without_removed = torch.where(
            nearest_slot[:, None, :] == removal_slot,
            second[:, None, :],
            nearest[:, None, :],
        )
        added = distance.gather(2, unselected[:, None, :].expand(-1, n, -1)).transpose(1, 2)
        costs = torch.minimum(without_removed[:, :, None, :], added[:, None, :, :]).sum(dim=-1)
        flat_cost, flat_index = costs.flatten(1).min(dim=1)
        current_cost = nearest.sum(dim=1)
        improve = flat_cost < current_cost - 1e-8
        if not bool(improve.any()):
            break
        remove = torch.div(flat_index, n - k, rounding_mode="floor")
        add_slot = flat_index.remainder(n - k)
        replacement = unselected.gather(1, add_slot[:, None]).squeeze(1)
        rows = torch.arange(batch, device=distance.device)[improve]
        selected[rows, remove[improve]] = replacement[improve]
    return selected


def medoid_objective(distance: Tensor, selected: Tensor) -> Tensor:
    return _selected_distances(distance, selected).min(dim=-1).values.sum(dim=-1)
