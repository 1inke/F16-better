"""Explicit in-memory workflow. Does not launch/read/write/fit on import.

Future runner: finish geometry_fold for ALL three folds before posthoc_fold.
External full source locks, resource guard, saving and independent audit required.
"""
import numpy as np
from core_neighbors import (make_bank, retrieve, calibration_weights, weighted_q95,
                            geometry_support, distance_components, label_support, best_positive, require)


def geometry_fold(train_z, train_ids, train_tracks, valid_z, valid_ids, valid_tracks, *, guard):
    require(callable(guard), "guard required")
    require(not np.intersect1d(train_ids, valid_ids).size, "train/valid IDs overlap")
    require(not np.intersect1d(train_tracks, valid_tracks).size, "train/valid tracks overlap")
    bank, query = make_bank(train_z, train_ids, train_tracks), make_bank(valid_z, valid_ids, valid_tracks)
    train_ix, train_d = retrieve(bank.z, bank.tracks, bank, guard=guard)
    weight = calibration_weights(train_tracks)
    radius = weighted_q95(train_d[:, -1], weight)
    ix, dist = retrieve(query.z, query.tracks, bank, guard=guard)
    guard()
    return dict(bank=bank, query=query, radius=radius, train_weights=weight,
                train_indices=train_ix, train_distances=train_d,
                train_distance_components=distance_components(bank.z, bank, train_ix),
                indices=ix, distances=dist, distance_components=distance_components(query.z, bank, ix),
                **geometry_support(dist, radius))


def posthoc_fold(geometry, train_labels):
    """Call only AFTER label-blind geometry is complete for every fold."""
    result = label_support(geometry["indices"], geometry["bank"], train_labels)
    result["supported"] = geometry["near"] & result["known_sufficient"]
    return result


def primary_rows(pairwise_scores, labels, all_candidate_fields):
    """Gather one old-R3 best-P query per known-P window AFTER OOF assembly.

    Caller first assembles every candidate's field in original ID order across
    all three folds. This function never searches, refits or chooses a neighbor.
    Check original IDs, [82,75,106] bins and [20,26,31] change bins in preflight.
    """
    chosen = best_positive(pairwise_scores, labels)
    rows = np.flatnonzero(chosen["hasP"])
    slots = chosen["slot"][rows]
    result = dict(window_index=rows, physical_slot=slots, rank=chosen["rank"][rows],
                  rank_bin=chosen["bin"][rows])
    for name, field in all_candidate_fields.items():
        a = np.asarray(field)
        require(a.shape[:2] == np.asarray(labels).shape, "all-20 OOF field shape: " + name)
        require(name not in result, "reserved primary field: " + name)
        result[name] = a[rows, slots]
    return result
