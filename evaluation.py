import numpy as np 
from Randomized_HOEDMD import conjugate_transpose
from scipy import linalg


def Evaluate_results_deter_pred(eigenvalues_deterministic,right_eigenvectors_deterministic,left_eigenvectors_deterministic,Phi_x_deterministic,eigenvalues,right_eigenvectors,left_eigenvectors_hat,Phi_x):
    """
    Compare deterministic and predicted modal analysis results by calculating errors.
    """
    # Sort deterministic eigenvalues (poles) in descending order
    sorted_lambda_s_indices_deterministic = np.argsort(eigenvalues_deterministic)[::-1]
    sorted_lambda_s_deterministic = eigenvalues_deterministic[sorted_lambda_s_indices_deterministic]

    # Compute deterministic modal amplitudes and reconstructed modes
    a_s_deterministic = conjugate_transpose(left_eigenvectors_deterministic) @ Phi_x_deterministic
    mode_deterministic = right_eigenvectors_deterministic * a_s_deterministic

    # Sort predicted eigenvalues in descending order
    sorted_lambda_s_indices_pred = np.argsort(eigenvalues)[::-1]
    sorted_lambda_s_pred = eigenvalues[sorted_lambda_s_indices_pred]
    
    # Compute predicted modal amplitudes and reconstructed modes
    a_s_pred = conjugate_transpose(left_eigenvectors_hat) @ Phi_x
    mode_pred = right_eigenvectors * a_s_pred

    # Sort both deterministic and predicted modes by eigenvalue order
    sorted_mode_deterministic = mode_deterministic[:, sorted_lambda_s_indices_deterministic]
    sorted_mode_pred = mode_pred[:, sorted_lambda_s_indices_pred]
    
    # Compute relative root mean square error (RRMSE) for poles and modes
    poles_error = calculate_rrmse_poles(sorted_lambda_s_deterministic, sorted_lambda_s_pred)
    modes_error = calculate_rrmse_modes(sorted_mode_deterministic, sorted_mode_pred)
    return poles_error, modes_error


def calculate_rrmse_modes(y_true, y_pred):
    """
    Calculate Relative Root Mean Square Error (RRMSE) between true and predicted modes.
    """
    # Compute numerator: RMSE for each column (mode)
    num = np.sqrt(np.mean(np.abs(y_true - y_pred) ** 2, axis=0))
    # Compute denominator: magnitude of true mode
    den = np.sqrt(np.sum(np.abs(y_true) ** 2, axis=0))
    # Compute RRMSE per mode and return the average
    rrmse = num / den
    return np.mean(rrmse)

def calculate_rrmse_poles(y_true, y_pred):
    """
    Calculate Relative Root Mean Square Error (RRMSE) between true and predicted eigenvalues.
    """
    rmse = np.sqrt(np.mean(np.abs(y_true - y_pred) ** 2))
    mean_true = np.sqrt(np.sum(np.abs(y_true) ** 2))
    # Avoid division by zero
    rrmse = rmse / mean_true if mean_true != 0 else np.nan
    return rrmse

def Evaluate_results(a_s,U_zhenshide,lambda_s,eigenvalues,right_eigenvectors,left_eigenvectors_hat,Phi_x0):
    """
    Evaluate DMD results by comparing predicted vs true modes and eigenvalues.
    """
    # Compute true modes weighted by their amplitudes
    mode_real = U_zhenshide * a_s
    
    # Sort true eigenvalues and corresponding modes
    sorted_lambda_s_indices_real = np.argsort(lambda_s)[::-1]
    sorted_lambda_s_real = lambda_s[sorted_lambda_s_indices_real]

    # Sort predicted eigenvalues and corresponding modes
    sorted_lambda_s_indices_pred = np.argsort(eigenvalues)[::-1]
    sorted_lambda_s_pred = eigenvalues[sorted_lambda_s_indices_pred]

    # Compute predicted modal amplitudes from left eigenvectors
    a_s_pred = conjugate_transpose(left_eigenvectors_hat) @ Phi_x0

    # Compute predicted modes using amplitudes
    mode_pred = right_eigenvectors * a_s_pred

    # Sort modes according to eigenvalue order
    sorted_mode_real = mode_real[:, sorted_lambda_s_indices_real]
    sorted_mode_pred = mode_pred[:, sorted_lambda_s_indices_pred]
    
    # Compute pole and mode RRMSE
    poles_error = calculate_rrmse_poles(sorted_lambda_s_real, sorted_lambda_s_pred)
    modes_error = calculate_rrmse_modes(sorted_mode_real, sorted_mode_pred)

    return poles_error, modes_error

def get_amplitudes(A, F):
    """Compute DMD amplitudes b from least-squares solution: F b = x1."""
    return linalg.lstsq(F, A[:, 0])[0]

def get_vandermonde(A, l):
    """Construct Vandermonde matrix using eigenvalues l."""
    return np.fliplr(np.vander(l, N=A.shape[1]))

def A_tilde(A, Fmodes, l):
    """
    Construct approximated data matrix using DMD modes and eigenvalues.
    A_tilde = Fmodes * diag(b) * V(λ)
    """
    # Compute amplitudes b
    b = get_amplitudes(A, Fmodes)
    # Compute Vandermonde matrix of eigenvalues
    V = get_vandermonde(A, l)
    # Reconstruct data using DMD model
    return Fmodes.dot(np.diag(b).dot(V))

def evaluate_rre_noHOEDMD(X_noisy,eigenvalues,right_eigenvectors):
    """
    Compute Relative Reconstruction Error (RRE) for standard DMD.
    """
    # Original noisy data
    real_matrix = X_noisy
    # Reconstructed data from DMD
    predicted_matrix = A_tilde(X_noisy, right_eigenvectors, eigenvalues)
    # Compute Frobenius norm of reconstruction error
    error = np.linalg.norm(real_matrix - predicted_matrix, 'fro')
    # Compute Frobenius norm of original data
    norm_real = np.linalg.norm(real_matrix, 'fro')
    # Return relative reconstruction error
    rre = error / norm_real
    return rre


def evaluate_rre(X_noisy,eigenvalues,right_eigenvectors,left_eigenvectors,Phi_x0,HOEDMD_flag):
    """
    Compute RRE for both standard and higher-order (HOEDMD) reconstructions.
    """
    # Original data
    real_matrix = X_noisy

    # If higher-order DMD is used, adjust amplitudes and modes
    if HOEDMD_flag:
        # Compute modal amplitudes
        a_s_pred = conjugate_transpose(left_eigenvectors) @ Phi_x0
        # Apply amplitudes to modes
        right_eigenvectors = right_eigenvectors * a_s_pred
        # Sort eigenvalues and modes
        sorted_lambda_s_indices_pred = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_lambda_s_indices_pred]
        right_eigenvectors = right_eigenvectors[:, sorted_lambda_s_indices_pred]

    # Compute reconstructed data
    predicted_matrix = A_tilde(X_noisy, right_eigenvectors, eigenvalues)
    # Compute reconstruction error
    error = np.linalg.norm(real_matrix - predicted_matrix, 'fro')
    # Compute normalization factor
    norm_real = np.linalg.norm(real_matrix, 'fro')
    # Return RRE
    rre = error / norm_real
    return rre


def evaluate_rre_zhenshide(X_noisy,lambda_s,U_zhenshide,a_s):
    """
    Compute RRE using true (known) eigenvalues, modes, and amplitudes.
    """
    # Original data matrix
    real_matrix = X_noisy

    # Construct full modes weighted by amplitudes
    right_eigenvectors = U_zhenshide * a_s

    # Sort eigenvalues and corresponding modes
    sorted_lambda_s_indices_pred = np.argsort(lambda_s)[::-1]
    eigenvalues = lambda_s[sorted_lambda_s_indices_pred]
    right_eigenvectors = right_eigenvectors[:, sorted_lambda_s_indices_pred]

    # Reconstruct data
    predicted_matrix = A_tilde(X_noisy, right_eigenvectors, eigenvalues)
    
    # Compute reconstruction and normalization norms
    error = np.linalg.norm(real_matrix - predicted_matrix, 'fro')
    norm_real = np.linalg.norm(real_matrix, 'fro')

    # Compute and return RRE
    rre = error / norm_real
    return rre
