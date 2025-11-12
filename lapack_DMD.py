import numpy as np

# Allowed mode options for DMD mode construction
_VALID_MODES = {'standard', 'exact', 'exact_scaled'}

def conjugate_transpose(M):
    """Return the conjugate transpose (Hermitian) of matrix M."""
    return np.conjugate(M).T

def compute_lapack_dmd(A, rank=None, dt=1, modes='exact', order=True):
    """Dynamic Mode Decomposition (article-aligned, pure NumPy).

    Parameters
    ----------
    A : array_like, shape (m, n)
        Snapshot matrix with columns as consecutive snapshots.
    rank : int or None
        If given, truncate to this rank (k <= n-1).
        If None, an automatic tolerance-based rank is chosen.
    dt : float or array_like, optional (default: 1)
        Time step between snapshots (broadcastable with np.log(l)).
    modes : {'standard','exact','exact_scaled'}
        - 'standard'     : F = U * W
        - 'exact'        : F = Y * V * (S^{-1}) * W
        - 'exact_scaled' : F = (1/lambda) * Y * V * (S^{-1}) * W
    order : bool
        If True, order modes/eigs by |omega| ascending.

    Returns
    -------
    F : ndarray, shape (m, k)
        Dynamic modes.
    l : ndarray, shape (k,)
        DMD eigenvalues (discrete-time).
    omega : ndarray, shape (k,)
        Continuous-time eigenvalues: log(l) / dt.
    """
    # ---- Input checks ----
    A = np.asarray_chkfinite(A)  # ensure ndarray with no inf/nan
    if modes not in _VALID_MODES:
        raise ValueError(f"modes must be one of {_VALID_MODES}, not {modes}")

    m, n = A.shape
    if n < 2:
        raise ValueError("A must have at least two columns (snapshots).")

    # rank must be in [1, n-1] if provided
    if rank is not None and (rank < 1 or rank > (n-1)):
        raise ValueError("rank must be >= 1 and <= n-1")

    # ---- Split snapshot sequences ----
    # X contains columns 0..n-2, Y contains columns 1..n-1
    X = A[:, :n-1]
    Y = A[:, 1:n]

    # ---- SVD of X (economy) ----
    # Thin SVD: X ≈ U diag(s) Vh, with shapes U(m,k), s(k,), Vh(k,n-1)
    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    # ---- Rank selection (automatic if not provided) ----
    if rank is None:
        # Heuristic tolerance based on leading singular value and machine eps
        eps = np.finfo(s.dtype).eps
        tau = (n-1) * eps * 10.0  # conservative multiplier
        k = int(np.sum(s >= tau * s[0])) if s.size else 0
        k = max(1, min(k, n-1))   # ensure 1 ≤ k ≤ n-1
    else:
        k = int(rank)

    # Truncate to rank k
    U = U[:, :k]
    s = s[:k]
    Vh = Vh[:k, :]

    # ---- Build reduced operator M = U^H Y V S^{-1} ----
    # Compute V = Vh^H; avoid forming diag(S^{-1}) until the end
    V = conjugate_transpose(Vh)
    # YV = Y @ V has shape (m, k)
    YV = Y @ V
    # Sinv = diag(1/s) has shape (k, k)
    Sinv = np.diag(1.0 / s)
    # G = Y V S^{-1} (m × k), a common intermediate for modes and M
    G = YV @ Sinv
    # Reduced operator M (k × k)
    M = conjugate_transpose(U) @ G

    # ---- Eigen-decomposition of reduced operator ----
    # l: discrete-time eigenvalues (λ), W: eigenvectors in reduced space
    l, W = np.linalg.eig(M)
    # Continuous-time eigenvalues ω = log(λ)/dt (supports scalar or array dt)
    omega = np.log(l) / dt

    # ---- Optional ordering by |omega| ----
    if order:
        sort_idx = np.argsort(np.abs(omega))
        l = l[sort_idx]
        omega = omega[sort_idx]
        W = W[:, sort_idx]

    # ---- Modes ----
    if modes == 'standard':
        # Standard DMD modes: F = U W (project reduced eigenvectors back)
        F = U @ W
    else:
        # Exact DMD modes (unscaled): F = Y V S^{-1} W = G W
        F = G @ W
        if modes == 'exact_scaled':
            # Scale each exact mode by 1/λ_i to match "exact_scaled" definition
            F = F / l

    return F, l, omega
