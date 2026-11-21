"""Pure Stage73 risk algebra, copied explicitly from Stage68 (no output imports).

Every contrast is risk(action a) - risk(action b), in radians squared. Native
softmax weights describe an empirical candidate measure, not a calibrated posterior.
"""
from __future__ import annotations

import hashlib
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import spearmanr

METHODS = ("FullPool", "Log2", "Mean3", "Matrix9_restore", "Matrix9_Taylor",
           "Matrix9_chordal", "Log2_diagonal", "Log2_isotropic")
GATES = (15, 29, 180)
SEED = 730701


def G(S):
    return np.trace(S) * np.eye(3) - S


def polar(M):
    U, _, Vh = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vh)]) @ Vh


def qscore(a, b, m, S):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return (np.sum(a * a - b * b, axis=-1) - 2 * (a - b) @ m
            - (np.einsum("ni,ij,nj->n", a, G(S), a)
               - np.einsum("ni,ij,nj->n", b, G(S), b)) / 12)


def moments(x, w):
    m = w @ x
    S = np.einsum("n,ni,nj->ij", w, x, x)
    M = np.einsum("n,nij->ij", w, Rot.from_rotvec(x).as_matrix())
    SM = M + M.T + (1 - np.trace(M)) * np.eye(3)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0],
                  M[1, 0] - M[0, 1]]) / 2
    mr = (np.eye(3) - G(SM) / 12) @ Rot.from_matrix(polar(M)).as_rotvec()
    return dict(m=m, S=S, M=M, SM=SM, v=v, mr=mr)


def evaluate(method, x, w, a, b, stats):
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    if method == "FullPool":
        Ra, Rb, pool = Rot.from_rotvec(a), Rot.from_rotvec(b), Rot.from_rotvec(x)
        return np.array([w @ ((aa.inv() * pool).magnitude() ** 2
                             - (bb.inv() * pool).magnitude() ** 2)
                         for aa, bb in zip(Ra, Rb)])
    m, S, M = stats["m"], stats["S"], stats["M"]
    if method == "Log2":
        return qscore(a, b, m, S)
    if method == "Mean3":
        return qscore(a, b, m, np.zeros((3, 3)))
    if method == "Matrix9_restore":
        return qscore(a, b, stats["mr"], stats["SM"])
    if method == "Matrix9_Taylor":
        return qscore(a, b, stats["v"], stats["SM"])
    if method == "Matrix9_chordal":
        return np.einsum("nij,ij->n", Rot.from_rotvec(b).as_matrix()
                         - Rot.from_rotvec(a).as_matrix(), M)
    if method == "Log2_diagonal":
        return qscore(a, b, m, np.diag(np.diag(S)))
    if method == "Log2_isotropic":
        return qscore(a, b, m, np.trace(S) * np.eye(3) / 3)
    raise ValueError(method)


def scores(x, w, a, b):
    stats = moments(x, w)
    vals = {method: evaluate(method, x, w, a, b, stats) for method in METHODS}
    return vals, stats["m"], stats["S"], stats["M"]


def queries(sample_id, gate, count=128):
    material = f"{sample_id}|{int(gate)}|{SEED}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    ab = rng.normal(size=(2, count, 3))
    ab /= np.linalg.norm(ab, axis=2, keepdims=True)
    ab *= rng.uniform(size=(2, count, 1)) ** (1 / 3) * np.deg2rad(min(gate, 29))
    return ab[0], ab[1], seed


def fidelity(vals):
    truth = vals["FullPool"]
    result = {}
    for method, value in vals.items():
        error = value - truth
        if np.ptp(truth) == 0 or np.ptp(value) == 0:
            rank = 1.0 if np.array_equal(value, truth) else None
        else:
            rank = float(spearmanr(value, truth).statistic)
        result[method] = dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                              mae=float(np.mean(abs(error))),
                              median_abs=float(np.median(abs(error))),
                              p95_abs=float(np.quantile(abs(error), .95)),
                              max_abs=float(np.max(abs(error))),
                              sign_agree=float(np.mean(np.sign(value) == np.sign(truth))),
                              rank_spearman=rank)
    return result


def payload_bytes(count):
    # Packed float32 state; reference rotation and common metadata are excluded.
    return dict(FullPool=16 * count, Log2=36, Mean3=12, Matrix9_restore=36,
                Matrix9_Taylor=36, Matrix9_chordal=36, Log2_diagonal=24,
                Log2_isotropic=16)


def timing(x, w, a, b, repeats=3):
    stats = moments(x, w)
    rows = []
    for count in (1, 8, 32, 128):
        for method in METHODS:
            elapsed = []
            for repeat in range(repeats + 1):
                tick = time.perf_counter_ns()
                evaluate(method, x, w, a[:count], b[:count], stats)
                duration = (time.perf_counter_ns() - tick) / 1000
                if repeat:
                    elapsed.append(duration)
            median = float(np.median(elapsed))
            rows.append(dict(method=method, queries=count, query_us=median,
                             query_us_per_pair=median / count))
    return rows


def numerical_sanity():
    """Checks independent trace distances, diagonal tensors and commuting rotations."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(12, 3)) * .25
    w = rng.uniform(size=12); w /= w.sum()
    a, b, _ = queries("numerical_sanity", 29)
    vals, _, _, _ = scores(x, w, a, b)
    pool = Rot.from_rotvec(x).as_matrix()
    def direct(v):
        actions = Rot.from_rotvec(v).as_matrix()
        cosines = (np.einsum("qij,pij->qp", actions, pool) - 1) / 2
        return np.arccos(np.clip(cosines, -1, 1)) ** 2 @ w
    trace_error = float(np.max(abs(vals["FullPool"] - (direct(a) - direct(b)))))
    assert trace_error < 5e-14, trace_error

    diagonal = np.array([.014, .023, .039])
    xd = np.concatenate([np.diag(np.sqrt(3 * diagonal)),
                         -np.diag(np.sqrt(3 * diagonal))])
    wd = np.ones(6) / 6
    vd, md, Sd, _ = scores(xd, wd, a, b)
    expected = np.sum((a * a - b * b)
                      * (1 - (diagonal.sum() - diagonal) / 12), axis=1)
    moment_error = float(np.max(abs(Sd - np.diag(diagonal))))
    diagonal_error = float(np.max(abs(vd["Log2_diagonal"] - expected)))
    assert np.max(abs(md)) < 1e-16 and moment_error < 1e-16
    assert diagonal_error < 2e-16
    assert np.max(abs(vd["Log2"] - vd["Log2_diagonal"])) < 1e-16
    isotropic = np.sum(a * a - b * b, axis=1) * (1 - diagonal.sum() / 18)
    isotropic_error = float(np.max(abs(vd["Log2_isotropic"] - isotropic)))
    assert isotropic_error < 2e-16

    xc = np.column_stack([np.array([-.3, .1, .25]), np.zeros((3, 2))])
    ac, bc = a.copy(), b.copy(); ac[:, 1:] = 0; bc[:, 1:] = 0
    vc, _, _, _ = scores(xc, np.array([.2, .3, .5]), ac, bc)
    commuting_error = float(np.max(abs(vc["Log2"] - vc["FullPool"])))
    assert commuting_error < 2e-15
    identical = scores(x, w, a, a)[0]
    assert all(np.max(abs(v)) < 1e-15 for v in identical.values())
    return dict(status="PASSED", seed=SEED,
                FullPool_direct_trace_max_abs=trace_error,
                diagonal_second_moment_max_abs=moment_error,
                diagonal_formula_max_abs=diagonal_error,
                isotropic_formula_max_abs=isotropic_error,
                commuting_Log2_FullPool_max_abs=commuting_error,
                identical_action_all_methods_zero=True, GT_pose_read=False)


if __name__ == "__main__":
    import json
    print(json.dumps(numerical_sanity(), indent=2))
