"""Test the DySO Python port on synthetic data with known structure."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import numpy as np
from dyso import dyso


def make_synthetic(seed=0):
    """Build two conditions sharing a 3D subspace and each having a 2D unique subspace.

    Total ambient dimensionality d=10. We construct:
      - shared signals (3 dims) that appear in BOTH conditions identically
      - A-unique signals (2 dims) that appear ONLY in condition A
      - B-unique signals (2 dims) that appear ONLY in condition B
      - 3 noise dims that should end up sharing/null indistinguishably
    """
    rng = np.random.default_rng(seed)
    d = 10
    T = 200  # samples per condition

    # Build a random orthonormal d×d basis; assign blocks to roles
    Q = np.linalg.qr(rng.standard_normal((d, d)))[0]
    U_shared = Q[:, :3]
    U_A = Q[:, 3:5]
    U_B = Q[:, 5:7]
    # Q[:, 7:] is "extra" — noise + unused

    # Time courses (smooth so PCA dims are well-defined)
    t = np.linspace(0, 4 * np.pi, T)
    shared_signals = np.stack(
        [np.sin(t), np.cos(0.7 * t), np.sin(0.3 * t + 0.5)], axis=1
    )
    A_signals = np.stack([np.sin(2.1 * t), np.cos(1.3 * t)], axis=1)
    B_signals = np.stack([np.cos(2.5 * t), np.sin(1.1 * t + 0.8)], axis=1)

    noise_scale = 0.05
    X_A = shared_signals @ U_shared.T + A_signals @ U_A.T + noise_scale * rng.standard_normal((T, d))
    X_B = shared_signals @ U_shared.T + B_signals @ U_B.T + noise_scale * rng.standard_normal((T, d))

    return X_A, X_B, dict(shared=U_shared, A=U_A, B=U_B)


def principal_angles(B1, B2):
    """Principal angles (in degrees) between two subspaces given by orthonormal bases."""
    if B1.shape[1] == 0 or B2.shape[1] == 0:
        return np.array([])
    U1, _ = np.linalg.qr(B1)
    U2, _ = np.linalg.qr(B2)
    s = np.linalg.svd(U1.T @ U2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def main():
    X_A, X_B, truth = make_synthetic(seed=42)

    print("=" * 60)
    print("Running DySO on synthetic data (d=10, 2 conditions)")
    print("Truth: shared=3D, A-unique=2D, B-unique=2D")
    print("=" * 60)

    result = dyso([X_A, X_B], var_cutoff=99.0, verbosity=0)

    A_basis = result.unique.get((0,), np.zeros((10, 0)))
    B_basis = result.unique.get((1,), np.zeros((10, 0)))
    S_basis = result.shared

    print("\n--- Recovered dimensionalities ---")
    print(f"A-unique: {A_basis.shape[1]}  (truth: 2)")
    print(f"B-unique: {B_basis.shape[1]}  (truth: 2)")
    print(f"Shared:   {S_basis.shape[1]}  (truth: 3, plus possibly noise dims)")

    print("\n--- Orthogonality check ---")
    full = result.full
    gram = full.T @ full
    err = np.max(np.abs(gram - np.eye(gram.shape[0])))
    print(f"||Q^T Q - I||_inf = {err:.2e}  (should be ~1e-10)")

    print("\n--- Principal angles vs ground truth (degrees) ---")
    print(f"A-unique recovered vs truth A: {principal_angles(A_basis, truth['A'])}")
    print(f"B-unique recovered vs truth B: {principal_angles(B_basis, truth['B'])}")
    # For shared, only check the first 3 recovered dims (the rest will be noise/extra)
    if S_basis.shape[1] >= 3:
        print(
            f"Shared (first 3 dims) vs truth shared: "
            f"{principal_angles(S_basis[:, :3], truth['shared'])}"
        )

    print("\n--- Crosstalk check ---")
    print(f"A-unique vs truth B (should be ~90°): {principal_angles(A_basis, truth['B'])}")
    print(f"B-unique vs truth A (should be ~90°): {principal_angles(B_basis, truth['A'])}")
    print(
        f"A-unique vs truth shared (should be ~90°): "
        f"{principal_angles(A_basis, truth['shared'])}"
    )

    print("\n--- Variance accounting ---")
    for cond, ve in result.var_explained.items():
        print(f"{cond}:")
        for k, v in ve.items():
            print(f"  {k}: {v:.1f}%")


if __name__ == "__main__":
    main()
