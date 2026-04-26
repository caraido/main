"""
DySO (Dynamic Subspace Overlap) — Python port of Dekleva et al. 2024.

Original MATLAB: https://github.com/pitt-rnel/action_imagery (BackEndHelpers/DySO.m)
Paper: Dekleva et al., Nat Hum Behav (2024). https://doi.org/10.1038/s41562-023-01804-5

Decomposes a list of (samples × dims) matrices, one per condition, into a single
orthonormal basis whose blocks are condition-unique and condition-shared.

Dependencies: numpy, pymanopt>=2.2
"""

from __future__ import annotations
from itertools import combinations
from dataclasses import dataclass

import numpy as np
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _pca_basis(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA via SVD on mean-centered X. Returns (components, explained_var_pct).

    Components are columns of shape (n_features, n_components).
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # Economy SVD on Xc gives Xc = U S Vt; components are rows of Vt.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt.T
    var = S ** 2 / max(Xc.shape[0] - 1, 1)
    ev_pct = 100.0 * var / var.sum()
    return components, ev_pct


def _potent_null_split(
    X_other: np.ndarray, var_cutoff: float, project_through: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Split the column space of X_other into potent (>= var_cutoff%) and null.

    Mirrors the two-step trick in DySO.m where the null-space dimensionality is
    matched in tail-variance terms when an outer projection is applied first.
    Returns (potent, null) bases as columns.
    """
    # Step 1: PCA on the raw not-i data
    pcs, ev = _pca_basis(X_other)
    cum = np.cumsum(ev)
    dim_potent = int(np.searchsorted(cum, var_cutoff) + 1)
    var_in_null = float(np.sum(np.var(X_other @ pcs[:, dim_potent:], axis=0)))

    if project_through is None or project_through.shape[1] == X_other.shape[1]:
        # Identity outer projection: just use the PCA split directly.
        return pcs[:, :dim_potent], pcs[:, dim_potent:]

    # Step 2: re-PCA in the projected coordinates and pick the null
    # dimensionality that matches the original tail variance.
    Xp = X_other @ project_through
    pcs2, _ = _pca_basis(Xp)
    proj_vars = np.var(Xp @ pcs2, axis=0)
    # Walk from the tail and count how many trailing dims accumulate <= var_in_null
    tail_cum = np.cumsum(proj_vars[::-1])
    dim_null = int(np.searchsorted(tail_cum, var_in_null, side="right"))
    dim_null = max(dim_null, 1)
    n_total = pcs2.shape[1]
    return pcs2[:, : n_total - dim_null], pcs2[:, n_total - dim_null :]


def _solve_stiefel(Z: np.ndarray, target: np.ndarray, k: int, verbosity: int = 0) -> np.ndarray:
    """Find Q on Stiefel(d, k) minimizing ||Z @ Q − target||_F^2."""
    d = Z.shape[1]
    manifold = Stiefel(d, k)

    @pymanopt.function.numpy(manifold)
    def cost(Q):
        R = Z @ Q - target
        return float(np.sum(R * R))

    @pymanopt.function.numpy(manifold)
    def egrad(Q):
        return 2.0 * Z.T @ (Z @ Q - target)

    # Quadratic cost ⇒ Hessian H(Q)[V] = 2 Z^T Z V (constant in Q).
    @pymanopt.function.numpy(manifold)
    def ehess(Q, V):
        return 2.0 * Z.T @ (Z @ V)

    problem = pymanopt.Problem(
        manifold=manifold,
        cost=cost,
        euclidean_gradient=egrad,
        euclidean_hessian=ehess,
    )
    optimizer = TrustRegions(verbosity=verbosity, max_iterations=1000)
    return optimizer.run(problem).point


# -----------------------------------------------------------------------------
# Core algorithm
# -----------------------------------------------------------------------------

@dataclass
class DySOResult:
    """Output of DySO.

    Attributes
    ----------
    unique : dict[tuple[int, ...], np.ndarray]
        Maps each combination of condition indices to its orthonormal basis.
        For the 2-condition case, keys are (0,) and (1,) — i.e., A-unique and B-unique.
        Higher combinations only appear if `combinations_mode='full'`.
    shared : np.ndarray
        Orthonormal basis (d × d_shared) for the fully-shared subspace.
    full : np.ndarray
        Orthonormal d × d basis: hstack of all unique blocks plus shared.
    var_explained : dict
        Per-condition variance fraction in each subspace (for sanity checks).
    """
    unique: dict[tuple[int, ...], np.ndarray]
    shared: np.ndarray
    full: np.ndarray
    var_explained: dict


def dyso(
    Xs: list[np.ndarray],
    var_cutoff: float = 99.0,
    combinations_mode: str = "single",
    verbosity: int = 0,
) -> DySOResult:
    """Decompose a list of condition matrices into unique + shared subspaces.

    Parameters
    ----------
    Xs : list of (n_samples_i, d) arrays
        One matrix per condition. All must share the same number of columns
        (the latent / channel dimensionality d). Trial-averaged data is typical.
    var_cutoff : float, default 99.0
        Percent variance cutoff used to delineate potent vs null subspaces.
    combinations_mode : {'single', 'full'}, default 'single'
        'single' identifies one unique subspace per condition (the paper's setting).
        'full' enumerates all combination lengths 1..N-1, peeling them off in order.
    verbosity : int
        Pymanopt verbosity for the trust-region solver.

    Returns
    -------
    DySOResult
    """
    Nc = len(Xs)
    if Nc < 2:
        raise ValueError("DySO needs at least 2 conditions.")
    d = Xs[0].shape[1]
    if any(X.shape[1] != d for X in Xs):
        raise ValueError("All conditions must share the same dimensionality.")

    # Drop rows with NaNs per condition
    Xs = [X[~np.isnan(X).any(axis=1)] for X in Xs]

    if combinations_mode == "single":
        comb_lengths = [1]
    elif combinations_mode == "full":
        comb_lengths = list(range(1, Nc))  # 1..N-1
    else:
        raise ValueError("combinations_mode must be 'single' or 'full'.")

    nuller = np.eye(d)  # running "previous-unique"-null projector
    unique: dict[tuple[int, ...], np.ndarray] = {}

    for z in comb_lengths:
        combs_z = list(combinations(range(Nc), z))
        # For each combination, estimate the form of its unique activity
        unique_forms = {}
        for comb in combs_z:
            not_in = [j for j in range(Nc) if j not in comb]
            X_not = np.vstack([Xs[j] for j in not_in])
            X_in = np.vstack([Xs[j] for j in comb])

            # Null space of not-`comb` (in the current "leftover" subspace)
            _, null_not = _potent_null_split(X_not, var_cutoff, project_through=nuller)

            # Project comb-data into nuller, then into the null space of not-comb.
            # That projection gives the FORM of comb-unique activity.
            X_in_null = X_in @ nuller @ null_not
            if X_in_null.shape[1] == 0 or X_in_null.shape[0] < 2:
                continue
            pcs_unique, _ = _pca_basis(X_in_null)

            # Decide how many dims of unique activity to keep (variance threshold)
            X_in_total_var = float(np.trace(np.cov(X_in.T))) if X_in.shape[1] > 1 else float(np.var(X_in))
            unique_var_pct = (
                np.var(X_in_null @ pcs_unique, axis=0) / max(X_in_total_var, 1e-12) * 100.0
            )
            # number of dims whose tail-cumulative variance crosses (100 - var_cutoff)
            tail_thr = 100.0 - var_cutoff
            n_keep = int(np.sum(np.cumsum(unique_var_pct[::-1]) >= tail_thr))
            n_keep = min(n_keep, pcs_unique.shape[1])
            if n_keep == 0:
                continue

            # Lift back to the full d-dim space
            U_form = nuller @ null_not @ pcs_unique[:, :n_keep]
            unique_forms[comb] = U_form

        if not unique_forms:
            if verbosity:
                print(f"No unique spaces of length {z}.")
            continue

        # --- Joint orthonormal optimization (the Manopt step) ---
        # Stack the per-comb unique forms and the corresponding "target" projections
        # of the concatenated data, then find a single Q on Stiefel(d_curr, K_total)
        # whose blocks reproduce each comb's projected activity.
        Z_curr = np.vstack(Xs) @ nuller  # data in the current leftover subspace
        d_curr = Z_curr.shape[1]

        block_sizes = [U.shape[1] for U in unique_forms.values()]
        K_total = sum(block_sizes)

        # Build the target: [Z_curr @ U_A, Z_curr @ U_B, ...] using forms re-expressed
        # in the leftover basis. nuller has orthonormal columns, so:
        # forms_in_leftover[k] satisfies (nuller @ forms_in_leftover[k]) == U_form
        # i.e. forms_in_leftover = nuller.T @ U_form
        targets = [Z_curr @ (nuller.T @ U) for U in unique_forms.values()]
        target = np.hstack(targets)

        Q = _solve_stiefel(Z_curr, target, K_total, verbosity=verbosity)

        # Slice Q into per-comb blocks and lift back to the original d-dim space
        offset = 0
        for comb, sz in zip(unique_forms.keys(), block_sizes):
            block = Q[:, offset : offset + sz]
            unique[comb] = nuller @ block  # back to original d-dim coords
            offset += sz

        # Update the leftover-null projector
        all_unique_so_far = np.hstack(list(unique.values()))
        nuller = _orth_complement(all_unique_so_far)

    # Shared subspace = orthogonal complement of all unique blocks
    if unique:
        all_unique = np.hstack(list(unique.values()))
        shared = _orth_complement(all_unique)
    else:
        shared = np.eye(d)

    # Tidy up: PCA-rotate shared and each unique block so the leading dims
    # carry the most variance (matches what the driver script does after DySO).
    X_all = np.vstack(Xs)
    shared = _pca_rotate(shared, X_all)
    for k in unique:
        unique[k] = _pca_rotate(unique[k], X_all)

    full = np.hstack(list(unique.values()) + [shared]) if unique else shared

    # Variance accounting (per-condition fractions in each subspace)
    var_explained = {}
    for ci, X in enumerate(Xs):
        total = float(np.sum(np.var(X, axis=0)))
        ve = {}
        for k, U in unique.items():
            ve[f"unique_{k}"] = float(np.sum(np.var(X @ U, axis=0))) / total * 100.0
        ve["shared"] = float(np.sum(np.var(X @ shared, axis=0))) / total * 100.0
        var_explained[f"cond_{ci}"] = ve

    return DySOResult(
        unique=unique, shared=shared, full=full, var_explained=var_explained
    )


def _orth_complement(B: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the orthogonal complement of col(B) in R^d."""
    d = B.shape[0]
    # SVD-based null space of B^T
    _, _, Vt = np.linalg.svd(B.T, full_matrices=True)
    rank = int(np.linalg.matrix_rank(B))
    return Vt[rank:].T  # columns span the complement


def _pca_rotate(basis: np.ndarray, X_all: np.ndarray) -> np.ndarray:
    """Right-rotate `basis` by PCA on the data projected into it.
    Keeps the basis orthonormal but reorders columns by variance."""
    if basis.shape[1] == 0:
        return basis
    Xb = X_all @ basis
    rot, _ = _pca_basis(Xb)
    return basis @ rot
