"""Pure posthoc classification. Selection comes from scores, never from GT."""
import numpy as np

LABELS = ["选中合格轨迹", "合格候选被漏选", "分离最优的指标错觉", "仅路径接近", "仅终点接近", "整体覆盖不足"]


def diagnose(paths, future, scores, ade_limit=.3, fde_limit=.5, dt=.4):
    paths, future, scores = map(np.asarray, (paths, future, scores))
    n, k, t, xy = paths.shape
    assert xy == 2 and future.shape == (n, t, 2) and scores.shape == (n, k)
    assert all(np.isfinite(x).all() for x in (paths, future, scores))
    assert ade_limit > 0 and fde_limit > 0 and dt > 0
    selected = np.argmax(scores, axis=1)
    row = np.arange(n)
    err = np.linalg.norm(paths.astype(np.float64) - future[:, None], axis=-1)
    ade, fde = err.mean(-1), err[..., -1]
    quality = ade + .5 * fde
    ia, iff, ij = [v.argmin(-1) for v in (ade, fde, quality)]
    good = (ade <= ade_limit) & (fde <= fde_limit)
    ag, fg = ade.min(-1) <= ade_limit, fde.min(-1) <= fde_limit
    selected_good, any_good = good[row, selected], good.any(-1)
    category = np.full(n, 5, dtype=np.int64)
    category[ag & ~fg] = 3
    category[fg & ~ag] = 4
    category[ag & fg & ~any_good] = 2
    category[any_good & ~selected_good] = 1
    category[selected_good] = 0
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, k + 1), order.shape), axis=1)
    prob = np.exp(scores - scores.max(-1, keepdims=True))
    prob /= prob.sum(-1, keepdims=True)
    pd = np.linalg.norm(paths[:, :, None] - paths[:, None], axis=-1).mean(-1)
    endpd = np.linalg.norm(paths[:, :, None, -1] - paths[:, None, :, -1], axis=-1)
    pairs = np.triu_indices(k, 1)
    velocity = np.diff(paths, axis=2, prepend=np.zeros((n, k, 1, 2))) / dt
    acceleration = np.diff(velocity, axis=2) / dt
    jerk = np.diff(acceleration, axis=2) / dt
    dominated = ((ade <= ade[row, selected, None] + 1e-9) & (fde <= fde[row, selected, None] + 1e-9)
                 & ((ade < ade[row, selected, None] - 1e-6) | (fde < fde[row, selected, None] - 1e-6))).any(-1)
    data = {
        "category": category, "selected": selected, "oracle_ade": ia, "oracle_fde": iff, "oracle_joint": ij,
        "ade_selected": ade[row, selected], "fde_selected": fde[row, selected],
        "minade": ade[row, ia], "minfde": fde[row, iff],
        "fde_at_minade": fde[row, ia], "ade_at_minfde": ade[row, iff],
        "ade_regret": ade[row, selected] - ade[row, ia], "fde_regret": fde[row, selected] - fde[row, iff],
        "joint_selected": quality[row, selected], "joint_oracle": quality[row, ij],
        "joint_regret": quality[row, selected] - quality[row, ij],
        "oracle_same_slot": ia == iff, "adequate_count": good.sum(-1),
        "selected_exact_ade_slot": selected == ia, "selected_exact_fde_slot": selected == iff,
        "selected_exact_joint_slot": selected == ij,
        "selected_ade_optimal_tolerance": ade[row, selected] - ade[row, ia] <= 1e-6,
        "selected_fde_optimal_tolerance": fde[row, selected] - fde[row, iff] <= 1e-6,
        "selected_pareto_dominated": dominated,
        "oracle_ade_score_rank": ranks[row, ia], "oracle_fde_score_rank": ranks[row, iff],
        "oracle_joint_score_rank": ranks[row, ij], "top_score_mass_uncalibrated": prob[row, selected],
        "endpoint_spread": endpd[:, pairs[0], pairs[1]].mean(-1),
        "duplicate_fraction_10cm": (pd[:, pairs[0], pairs[1]] < .1).mean(-1),
        "selected_speed_p95": np.quantile(np.linalg.norm(velocity[row, selected], axis=-1), .95, axis=-1),
        "selected_acceleration_mean": np.linalg.norm(acceleration[row, selected], axis=-1).mean(-1),
        "selected_jerk_mean": np.linalg.norm(jerk[row, selected], axis=-1).mean(-1),
    }
    assert np.all(data["ade_regret"] >= -1e-10) and np.all(data["fde_regret"] >= -1e-10)
    assert np.all(data["joint_regret"] >= -1e-10)
    assert np.array_equal(category == 0, selected_good)
    assert np.array_equal(np.isin(category, [0, 1]), any_good)
    return data, ade, fde
