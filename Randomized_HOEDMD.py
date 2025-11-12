import numpy as np
from scipy import linalg
from scipy.linalg import solve
import numbers

def random_gaussian_map(A, l, axis, random_state):
    """Generate a Gaussian random projection matrix."""
    # Create a random matrix with normally distributed entries (mean 0, variance 1)
    # Shape depends on the selected axis of A
    return random_state.standard_normal(size=(A.shape[axis], l)).astype(A.dtype)

def check_random_state(seed):
    """Convert input seed into a numpy RandomState instance."""
    # If seed is None, return global RandomState
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    # If seed is an integer, create new RandomState with this seed
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    # If seed is already a RandomState, return it as is
    if isinstance(seed, np.random.RandomState):
        return seed
    # Otherwise, raise an error
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def Inverse_lower_triangular(Matrix):
    """Compute the inverse of a lower triangular matrix efficiently."""
    # Create identity matrix of the same size
    I = np.eye(Matrix.shape[0])
    # Use forward substitution to solve for the inverse (faster than general inversion)
    Matrix_inv = solve(Matrix, I)
    return Matrix_inv

def orthonormalize_cholesky(A, overwrite_a=True, check_finite=False):
    """Orthonormalize columns of A using Cholesky decomposition."""
    # Compute Cholesky factorization (A = L * L^T)
    L = linalg.cholesky(A, lower=True, overwrite_a=overwrite_a)
    return L

def perform_subspace_iterations_cholesky(A, Y, n_iter=2, axis=1):
    """Perform subspace iterations using Cholesky-based orthonormalization."""
    # Iteratively refine Y to better capture the range of A
    for _ in range(n_iter):
        # Compute intermediate matrices
        Mxy = conjugate_transpose(A) @ Y
        Myy = conjugate_transpose(Y) @ Y

        # Compute Cholesky decomposition of Myy
        L1 = orthonormalize_cholesky(Myy)

        # Compute intermediate matrix Ml
        Ml = Mxy @ Inverse_lower_triangular(conjugate_transpose(L1))

        if axis == 1:
            # Apply another Cholesky orthonormalization
            L2 = orthonormalize_cholesky(conjugate_transpose(Ml) @ Ml)

            # Update Y with refined subspace
            Y = A @ Ml @ Inverse_lower_triangular(conjugate_transpose(L2))

    return Y

def johnson_lindenstrauss(A, l, axis=1, random_state=None):
    """Perform the Johnson-Lindenstrauss random projection on matrix A."""
    random_state = check_random_state(random_state)

    A = np.asarray(A)
    # Ensure A is 2D
    if A.ndim != 2:
        raise ValueError('A must be a 2D array, not %dD' % A.ndim)
    if axis not in (0, 1):
        raise ValueError('If supplied, axis must be in (0, 1)')

    # Generate Gaussian random projection matrix
    Omega = random_gaussian_map(A, l, axis, random_state)

    # Project A onto random subspace
    if axis == 0:
        return Omega.T.dot(A)
    return A.dot(Omega)

def generate_phi_matrices(data, p):
    """Generate time-delay embedded matrices for Higher-Order DMD."""
    # Input: data (M x N), p = embedding order
    # Output: list of (p+1) delayed snapshot matrices, each of shape (M, N-p)
    M, N = data.shape
    phi = []
    for i in range(p + 1):
        if i + (N - p) <= N:
            phi_i = data[:, i:i + (N - p)]
            phi.append(phi_i)
    return phi

def conjugate_transpose(A):
    """Return the conjugate transpose (Hermitian) of A."""
    if A.dtype == np.complex128 or A.dtype == np.complex64:
        return A.conj().T
    return A.T

def HOEDMD(data, p, S, block=False):
    """Perform Higher Order Extended Dynamic Mode Decomposition (HOEDMD)."""
    # Get data dimensions
    M = data.shape[0]  # spatial dimension
    N = data.shape[1]  # temporal dimension

    # Step 0: Construct time-delay embedded matrix
    phi_list = generate_phi_matrices(data, p)
    Phi = np.concatenate(phi_list, axis=0)
    Phi_x0 = np.concatenate(phi_list[:-1], axis=0)[:, 0]

    # Step 1: Compute SVD and apply rank-S truncation
    if block:
        import dask.array as da
        Phi_dask = da.from_array(Phi, chunks=(500, N))
        U_dask, s_dask, vt_dask = da.linalg.svd(Phi_dask)
        U = U_dask.compute()
    else:
        U, Sum, _ = np.linalg.svd(Phi, full_matrices=False)
    P = U[:, :S]

    # Step 2: Split projection matrix into components
    P_x = P[:(M*p), :]
    P_y = P[M:, :]
    P_1 = P[M:(2*M), :]

    # Compute SVD of P_x
    M_Px, Omega_Px, NH_Px = np.linalg.svd(P_x, full_matrices=False)

    # Step 3: Compute approximate DMD operator
    N_Px = conjugate_transpose(NH_Px)
    MH_Px = conjugate_transpose(M_Px)
    eigenvalues, right_eigenvectors_tilde = np.linalg.eig(
        N_Px @ np.linalg.inv(np.diag(Omega_Px)) @ MH_Px @ P_y
    )
    left_eigenvectors_tilde = conjugate_transpose(np.linalg.inv(right_eigenvectors_tilde))

    # Step 4: Map eigenvectors back to original space
    right_eigenvectors = P_1 @ right_eigenvectors_tilde
    left_eigenvectors_hat = (
        M_Px @ np.linalg.inv(np.diag(Omega_Px)) @ NH_Px @ left_eigenvectors_tilde
    )

    # Step 5: Normalize eigenvectors
    norms = np.linalg.norm(right_eigenvectors, axis=0)
    fenmu = []
    for i in range(S):
        fenmu.append(
            np.conjugate(eigenvalues[i])
            * conjugate_transpose(right_eigenvectors_tilde[:, i].reshape(-1, 1))
            @ left_eigenvectors_tilde[:, i].reshape(-1, 1)
        )
    fenmu = np.array(fenmu).reshape(S,)

    right_eigenvectors = right_eigenvectors / norms
    left_eigenvectors_hat = left_eigenvectors_hat * norms / fenmu

    return eigenvalues, right_eigenvectors, left_eigenvectors_hat, Phi_x0

def Randomized_HOEDMD(X, S, p, oversample=10, n_subspace=2, random_state=None, block=False):
    """Perform Randomized Higher Order Extended Dynamic Mode Decomposition."""
    # Step 1: Random projection using Johnson–Lindenstrauss transform
    Y = johnson_lindenstrauss(X, S + oversample, random_state=random_state)

    # Step 2: Optional subspace iterations for improved accuracy
    if n_subspace > 0:
        Y = perform_subspace_iterations_cholesky(X, Y, n_iter=n_subspace, axis=1)

    # Step 3: Orthonormalize Y using Cholesky decomposition
    L3 = orthonormalize_cholesky(conjugate_transpose(Y) @ Y)

    # Step 4: Perform HOEDMD on the reduced matrix
    eigenvalues, right_eigenvectors, left_eigenvectors_hat, Phi_x0 = HOEDMD(
        Inverse_lower_triangular(L3) @ conjugate_transpose(Y) @ X,
        p=p,
        S=S,
        block=block,
    )

    # Step 5: Map eigenvectors back to original high-dimensional space
    right_eigenvectors = (
        Y @ conjugate_transpose(Inverse_lower_triangular(L3)) @ right_eigenvectors
    )

    return eigenvalues, right_eigenvectors, left_eigenvectors_hat, Phi_x0








