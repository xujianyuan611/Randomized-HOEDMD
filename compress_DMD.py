import numpy as np

# Define valid DMD mode computation methods
_VALID_MODES = {'standard', 'exact', 'exact_scaled'}

def conjugate_transpose(M):
    """Return the conjugate transpose (Hermitian transpose) of matrix M."""
    return np.conjugate(M).T

def _column_scale(X, Y):
    """Normalize (X, Y) column-wise by their 2-norms to preserve the equivalence class [A] of AX=Y."""
    d = np.linalg.norm(X, axis=0)       # Compute column-wise norms of X
    d[d == 0.0] = 1.0                   # Avoid division by zero for zero-norm columns
    invd = 1.0 / d                      # Compute reciprocal of norms
    return X * invd, Y * invd           # Scale both X and Y columns by invd

def compute_dmd_compress(A, rank=None, dt=1, modes='exact', order=True):
    """
    Compute Compressed / QR-compressed Dynamic Mode Decomposition (xGEDMDQ, Algorithm 3).

    Parameters
    ----------
    A : ndarray, shape (n_state, n_snaps=m+1)
        Snapshot matrix F containing consecutive system states z1,...,zm,zm+1.
    rank : int or None
        Target rank for truncation. If None, tolerance-based truncation is used.
    dt : float
        Time step between snapshots.
    modes : {'standard','exact','exact_scaled'}
        Specifies how dynamic modes are computed:
            - 'standard'     : F = Qhat * (U_k * W_k)
            - 'exact'        : F = Qhat * (Ry * V_k * S_k^{-1} * W_k)
            - 'exact_scaled' : F = Qhat * ((Ry * V_k * S_k^{-1} * W_k) / λ_i)
    order : bool
        If True, sort modes and eigenvalues by increasing |ω|.

    Returns
    -------
    F : ndarray (n_state, k)
        Dynamic modes in the original state space.
    l : ndarray (k,)
        Discrete-time eigenvalues.
    omega : ndarray (k,)
        Continuous-time eigenvalues, computed as log(l)/dt.
    """
    # -------- Step 0: Input validation --------
    A = np.asarray_chkfinite(A)  # Ensure finite numerical values
    if modes not in _VALID_MODES:
        raise ValueError(f"modes must be one of {_VALID_MODES}, not {modes}")
    n_state, n_snaps = A.shape
    if n_snaps < 2:
        raise ValueError("Need at least two snapshots (columns) in A.")

    # ===== Step 1: QR factorization of F =====
    # Perform thin QR decomposition: A = Qhat * R
    # Qhat (n_state × r) forms an orthonormal basis, R (r × n_snaps) contains projection coefficients
    Qhat, R = np.linalg.qr(A, mode='reduced')

    # ===== Step 2: Build Rx and Ry in compressed coordinates =====
    r = R.shape[0]               # Dimension of reduced subspace
    Rx = R[:, :n_snaps - 1]      # Matrix of past snapshots (R[:,1:m])
    Ry = R[:, 1:n_snaps]         # Matrix of future snapshots (R[:,2:m+1])

    # Optionally perform column normalization (recommended for numerical stability)
    Rx, Ry = _column_scale(Rx, Ry)

    # ===== Step 3: Apply xGEDMD (Algorithm 2) in compressed space =====
    # Compute SVD of Rx (economy version)
    U, s, Vh = np.linalg.svd(Rx, full_matrices=False)

    # ===== Step 4: Determine truncation rank =====
    if rank is None:
        # Automatic tolerance-based rank selection
        eps = np.finfo(s.dtype).eps
        tau = (Rx.shape[1]) * eps * 10.0  # Conservative threshold factor
        k = int(np.sum(s >= tau * s[0])) if s.size else 0
        k = max(1, min(k, Rx.shape[1]))   # Ensure at least rank-1
    else:
        if not (1 <= rank <= Rx.shape[1]):
            raise ValueError("rank must be in [1, n_snaps-1]")
        k = int(rank)

    # Truncate to selected rank
    U = U[:, :k]            # (r × k)
    s = s[:k]               # (k,)
    Vh = Vh[:k, :]          # (k × r)
    V = conjugate_transpose(Vh)      # Compute V = Vh^H
    Sinv = np.diag(1.0 / s)          # Invert singular values

    # ===== Step 5: Compute reduced operator in compressed space =====
    # Gc = Ry * V * S^{-1}
    Gc = (Ry @ V) @ Sinv             # (r × k)
    # M = U^H * Gc  (low-dimensional system matrix)
    M = conjugate_transpose(U) @ Gc  # (k × k)

    # ===== Step 6: Eigen-decomposition =====
    # Compute discrete-time eigenvalues and eigenvectors
    l, W = np.linalg.eig(M)
    # Convert to continuous-time eigenvalues ω = log(λ)/Δt
    omega = np.log(l) / dt

    # ===== Step 7: Optional sorting by eigenvalue magnitude =====
    if order:
        idx = np.argsort(np.abs(omega))
        l = l[idx]
        omega = omega[idx]
        W = W[:, idx]

    # ===== Step 8: Reconstruct modes in original space =====
    if modes == 'standard':
        # Standard DMD: F = Qhat * (U * W)
        Zhat = U @ W
        F = Qhat @ Zhat
    else:
        # Exact DMD: F = Qhat * (Ry * V * S^{-1} * W)
        Zhat_ex = Gc @ W
        F = Qhat @ Zhat_ex
        if modes == 'exact_scaled':
            # Scale each mode by 1/λ_i to match "exact_scaled" formulation
            scale = np.ones_like(l, dtype=complex)
            nz = (l != 0)
            scale[nz] = 1.0 / l[nz]
            F = F * scale  # Apply scaling per mode (column-wise)

    return F, l, omega
