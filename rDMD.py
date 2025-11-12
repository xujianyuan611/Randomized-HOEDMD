import numpy as np
from Randomized_HOEDMD import conjugate_transpose, check_random_state, johnson_lindenstrauss
from scipy import linalg
from math import log, sqrt
from scipy import sparse
from functools import partial

# Valid mode options for Dynamic Mode Decomposition (DMD)
_VALID_MODES = ('standard', 'exact', 'exact_scaled')

def orthonormalize(A, overwrite_a=True, check_finite=False):
    """Orthonormalize the columns of matrix A using QR decomposition."""
    # For matrix A of shape (m, n), 'economic' mode returns Q(m, k), R(k, n),
    # where k = min(m, n). This avoids unnecessary computations for large matrices.
    Q, _ = linalg.qr(A, overwrite_a=overwrite_a, check_finite=check_finite,
                     mode='economic', pivoting=False)
    return Q

def perform_subspace_iterations(A, Q, n_iter=2, axis=1):
    """Perform subspace iteration on Q to improve approximation of A's range."""
    # If axis=0, transpose Q to work row-wise.
    if axis == 0:
        Q = Q.T

    # Start by orthonormalizing Q
    Q = orthonormalize(Q)

    # Perform n_iter iterations to refine the subspace
    for _ in range(n_iter):
        if axis == 0:
            Z = orthonormalize(A.dot(Q))
            Q = orthonormalize(A.T.dot(Z))
        else:
            Z = orthonormalize(A.T.dot(Q))
            Q = orthonormalize(A.dot(Z))

    # Transpose back if we worked row-wise
    if axis == 0:
        return Q.T
    return Q

def safe_sparse_dot(a, b, *, dense_output=False):
    """Compute a dot product that safely handles sparse matrices."""
    # This function extends np.dot or @ to support sparse matrices and higher dimensions.
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # Sparse matrices are always 2D. If a is sparse and b is high-dimensional,
            # reshape b to perform multiplication.
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # If b is sparse, flatten a accordingly.
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    # Convert to dense array if required
    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret

def sparse_random_map(A, l, axis, density, random_state):
    """Generate a sparse random projection matrix for dimensionality reduction."""
    # The random entries are ±sqrt(1/density), chosen with equal probability.
    values = (-sqrt(1. / density), sqrt(1. / density))
    data_rvs = partial(random_state.choice, values)

    return sparse.random(A.shape[axis], l, density=density, data_rvs=data_rvs,
                         random_state=random_state, dtype=A.dtype)

def sparse_johnson_lindenstrauss(A, l, density=None, axis=1, random_state=None):
    """
    Compute a sparse Johnson–Lindenstrauss random projection of A.

    Given an m x n matrix A, returns an m x l (or l x n) orthonormal matrix Q
    that approximates the range of A.
    """
    random_state = check_random_state(random_state)
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError('A must be a 2D array')

    if axis not in (0, 1):
        raise ValueError('Axis must be 0 or 1')

    # Default sparsity based on matrix size
    if density is None:
        density = log(A.shape[0]) / A.shape[0]

    # Generate sparse random projection matrix
    Omega = sparse_random_map(A, l, axis, density, random_state)

    # Project A using Omega
    if axis == 0:
        return safe_sparse_dot(Omega.T, A)
    return safe_sparse_dot(A, Omega)

def _compute_rqb(A, rank, oversample, n_subspace, sparse, random_state):
    """Compute the randomized QB decomposition of A."""
    # Use sparse or dense random projection depending on 'sparse' flag
    if sparse:
        Q = sparse_johnson_lindenstrauss(A, rank + oversample,
                                         random_state=random_state)
    else:
        Q = johnson_lindenstrauss(A, rank + oversample, random_state=random_state)

    # Optionally perform subspace iterations for improved accuracy
    if n_subspace > 0:
        Q = perform_subspace_iterations(A, Q, n_iter=n_subspace, axis=1)
    else:
        Q = orthonormalize(Q)

    # Project A onto the subspace
    B = conjugate_transpose(Q).dot(A)
    return Q, B

def compute_rqb(A, rank, oversample=20, n_subspace=2, n_blocks=1, sparse=False,
                random_state=None):
    """Compute a (possibly block-wise) randomized QB decomposition."""
    if n_blocks > 1:
        m, n = A.shape
        # Split A into n_blocks of rows
        row_sets = np.array_split(range(m), n_blocks)
        Q_block, K = [], []

        for rows in row_sets:
            Qtemp, Ktemp = _compute_rqb(np.asarray_chkfinite(A[rows, :]), 
                rank=rank, oversample=oversample, n_subspace=n_subspace, 
                sparse=sparse, random_state=random_state)
            Q_block.append(Qtemp)
            K.append(Ktemp)

        # Combine results across blocks
        Q_small, B = _compute_rqb(
            np.concatenate(K, axis=0), rank=rank, oversample=oversample,
            n_subspace=n_subspace, sparse=sparse, random_state=random_state)

        Q_small = np.vsplit(Q_small, n_blocks)
        Q = [Q_block[i].dot(Q_small[i]) for i in range(n_blocks)]
        Q = np.concatenate(Q, axis=0)
    else:
        Q, B = _compute_rqb(np.asarray_chkfinite(A), 
            rank=rank, oversample=oversample, n_subspace=n_subspace,
            sparse=sparse, random_state=random_state)

    return Q, B

def compute_dmd(A, rank=None, dt=1, modes='exact', order=True):
    """Perform standard Dynamic Mode Decomposition (DMD)."""
    # DMD decomposes A into spatial modes and temporal dynamics.
    A = np.asarray_chkfinite(A)
    m, n = A.shape

    if modes not in _VALID_MODES:
        raise ValueError('Invalid mode type')

    if rank is not None and (rank < 1 or rank > n):
        raise ValueError('Invalid rank value')

    # Split A into time-shifted snapshots
    X = A[:, :(n-1)]
    Y = A[:, 1:n]

    # Compute SVD of X
    U, s, Vh = linalg.svd(X, compute_uv=True, full_matrices=False)

    if rank is not None:
        U = U[:, :rank]
        s = s[:rank]
        Vh = Vh[:rank, :]

    # Compute low-dimensional system matrix M
    G = np.dot(Y, conjugate_transpose(Vh)) / s
    M = np.dot(conjugate_transpose(U), G)

    # Eigen decomposition of M gives dynamics
    l, W = linalg.eig(M, right=True, overwrite_a=True)
    omega = np.log(l) / dt

    # Sort modes by magnitude
    if order:
        sort_idx = np.argsort(np.abs(omega))
        W = W[:, sort_idx]
        l = l[sort_idx]
        omega = omega[sort_idx]

    # Compute DMD modes
    if modes == 'standard':
        F = np.dot(U, W)
    else:
        F = np.dot(G, W)
        if modes == 'exact_scaled':
            F /= l

    return F, l, omega

def compute_rdmd(A, rank, dt=1, oversample=10, n_subspace=2, modes='standard',
                 order=True, random_state=None):
    """Perform Randomized Dynamic Mode Decomposition (rDMD)."""
    # Step 1: Compute randomized QB decomposition
    Q, B = compute_rqb(A, rank, oversample=oversample, n_subspace=n_subspace,
                       random_state=random_state)

    # Step 2: Apply standard DMD on reduced matrix B
    F, l, omega = compute_dmd(B, rank=rank, dt=dt, modes=modes, order=order)

    # Step 3: Project modes back to original space
    F = Q.dot(F)

    return F, l, omega