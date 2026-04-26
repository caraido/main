"""Test the N=3 generalization of the DySO port.

Setup mimics a 3-way variance-partitioning scenario (e.g., DINOv2 / CLIP / GloVe
RDMs viewed as conditions): each condition has its own unique signal plus all
three share a common subspace.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from dyso import dyso
from test_dyso import principal_angles


def make_three_condition(seed=1):
    rng = np.random.default_rng(seed)
    d = 12
    T = 250

    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    U_shared = Q[:, :3]
    U_A = Q[:, 3:5]
    U_B = Q[:, 5:7]
    U_C = Q[:, 7:9]

    t = np.linspace(0, 4 * np.pi, T)
    shared_sig = np.stack([np.sin(t), np.cos(0.7 * t), np.sin(0.3 * t)], axis=1)
    A_sig = np.stack([np.sin(2.1 * t), np.cos(1.3 * t)], axis=1)
    B_sig = np.stack([np.cos(2.5 * t), np.sin(1.1 * t + 0.8)], axis=1)
    C_sig = np.stack([np.sin(1.7 * t + 0.3), np.cos(2.2 * t)], axis=1)

    noise = 0.05
    X_A = shared_sig @ U_shared.T + A_sig @ U_A.T + noise * rng.standard_normal((T, d))
    X_B = shared_sig @ U_shared.T + B_sig @ U_B.T + noise * rng.standard_normal((T, d))
    X_C = shared_sig @ U_shared.T + C_sig @ U_C.T + noise * rng.standard_normal((T, d))

    return [X_A, X_B, X_C], dict(shared=U_shared, A=U_A, B=U_B, C=U_C)


def main():
    Xs, truth = make_three_condition()

    print("=" * 60)
    print("DySO on 3 conditions, mode='single' (per-condition unique only)")
    print("=" * 60)
    res = dyso(Xs, var_cutoff=99.0, combinations_mode="single")

    print("\n--- Recovered dims ---")
    for k, U in res.unique.items():
        print(f"  unique_{k}: {U.shape[1]}D")
    print(f"  shared:    {res.shared.shape[1]}D")

    print("\n--- Principal angles vs ground truth (deg, mean) ---")
    for k, name in [((0,), "A"), ((1,), "B"), ((2,), "C")]:
        ang = principal_angles(res.unique[k], truth[name])
        print(f"  unique_{k} vs truth {name}: {ang} (mean={ang.mean():.2f}°)")
    if res.shared.shape[1] >= 3:
        ang = principal_angles(res.shared[:, :3], truth["shared"])
        print(f"  shared (top-3) vs truth shared: {ang} (mean={ang.mean():.2f}°)")

    print("\n--- Per-condition variance accounting (%) ---")
    for cond, ve in res.var_explained.items():
        items = ", ".join(f"{k}={v:.1f}" for k, v in ve.items())
        print(f"  {cond}: {items}")


if __name__ == "__main__":
    main()
