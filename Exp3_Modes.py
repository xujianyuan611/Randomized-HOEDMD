import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from Randomized_HOEDMD import conjugate_transpose, HOEDMD, Randomized_HOEDMD
from evaluation import Evaluate_results, evaluate_rre, Evaluate_results_deter_pred, calculate_rrmse_poles, calculate_rrmse_modes
from datetime import datetime
import time
import scipy.io
from rDMD import compute_dmd, compute_rdmd
from lapack_DMD import compute_lapack_dmd
from compress_DMD import compute_dmd_compress

import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as sio


def create_timestamped_run_folder():
    """
    Create a timestamped folder for saving experiment results.

    Returns:
        str: The full path of the created folder.
             Format: "<script_name>_runs/YYYYMMDD_HHMMSS/"

    Example:
        >>> folder = create_timestamped_run_folder()
        >>> print(folder)
        'my_script_runs/20230815_143022/'
    """
    # Get the current script name (without ".py" extension)
    script_name = os.path.basename(__file__).replace('.py', '')
    
    # Create a timestamp string in format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Combine script name and timestamp to create a folder name
    folder_path = f"{script_name}_runs/{timestamp}"
    
    # Create the directory, including parent directories if they do not exist
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path


def generate_gaussian_noise(X_noisy):
    """
    Generate additive Gaussian noise for a given input signal matrix.

    Args:
        X_noisy (ndarray): Input signal matrix.

    Returns:
        ndarray: Gaussian noise matrix with target SNR of 10 dB.
    """
    SNR_dB = 10                                # Target signal-to-noise ratio (in dB)
    signal_power = np.mean(X_noisy ** 2)       # Calculate signal power
    SNR_linear = 10 ** (SNR_dB / 10)           # Convert dB to linear scale
    noise_power = signal_power / SNR_linear    # Compute required noise power
    noise = np.random.normal(0, np.sqrt(noise_power), X_noisy.shape)  # Generate Gaussian noise
    return noise


def save_modes(modes, method_name, grid_shape, folder_path):
    """
    Plot and save the first few DMD modes as contour plots.

    Args:
        modes (ndarray): Mode matrix (flattened spatial data per column).
        method_name (str): Name of the DMD variant (used in file naming).
        grid_shape (tuple): Shape to reshape each mode into 2D grid.
        folder_path (str): Folder path to save the plot.
    """
    # Create 2x3 subplots with tight vertical spacing
    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                             gridspec_kw={'wspace': 0.1, 'hspace': -0.3})
    axes = axes.ravel()

    modes_to_plot = 6  # Number of modes to visualize
    
    for i in range(modes_to_plot):
        mode = modes[:, i].reshape(grid_shape)
        ax = axes[i]
        c = ax.contourf(mode.T, levels=50, cmap='viridis')  # Plot mode as filled contour
        ax.set_aspect('equal')
        
        # Hide axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Add a filled black circle at a fixed position (for reference)
        circle = plt.Circle((50, 100), 25, color='black', fill=True)
        ax.add_patch(circle)
    
    # Save the figure to the specified folder
    file_name = f'{method_name}_DMD_Modes.png'
    plt.savefig(os.path.join(folder_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_and_save_eigenvalues_with_unit_circle(l_det, l_rand, l_det1, l_rand1, l_compress_DMD, folder_path):
    """
    Plot DMD eigenvalues for multiple methods on the complex plane with a unit circle.

    Args:
        l_det (ndarray): Deterministic HOEDMD eigenvalues.
        l_rand (ndarray): Randomized HOEDMD eigenvalues.
        l_det1 (ndarray): Deterministic standard DMD eigenvalues.
        l_rand1 (ndarray): Randomized DMD eigenvalues.
        l_compress_DMD (ndarray): Compressed DMD eigenvalues.
        folder_path (str): Folder to save the resulting figure.
    """
    # Extract real and imaginary parts of eigenvalues
    eigvals_det_real = l_det.real
    eigvals_det_imag = l_det.imag
    eigvals_rand_real = l_rand.real
    eigvals_rand_imag = l_rand.imag
    eigvals_det_real1 = l_det1.real
    eigvals_det_imag1 = l_det1.imag
    eigvals_rand_real1 = l_rand1.real
    eigvals_rand_imag1 = l_rand1.imag
    eigvals_comp_real1 = l_compress_DMD.real
    eigvals_comp_imag1 = l_compress_DMD.imag

    # Initialize figure
    plt.figure(figsize=(7, 7))

    # Plot eigenvalues from each method using different markers and colors
    plt.scatter(eigvals_det_real, eigvals_det_imag, color='b', label='Det HOEDMD', alpha=0.7, marker='s', s=100)
    plt.scatter(eigvals_rand_real, eigvals_rand_imag, color='r', label='Rand HOEDMD', alpha=0.7, marker='*', s=150)
    plt.scatter(eigvals_det_real1, eigvals_det_imag1, color='yellow', label='Det DMD', alpha=0.7, marker='^', s=140)
    plt.scatter(eigvals_rand_real1, eigvals_rand_imag1, color='g', label='Rand DMD', alpha=0.7, marker='o', s=60)
    plt.scatter(eigvals_comp_real1, eigvals_comp_imag1, color='#DDA0DD', label='Comp DMD', alpha=0.7, marker='d', s=50)

    # Draw unit circle for stability reference
    unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5)
    plt.gca().add_artist(unit_circle)

    # Set labels and grid
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()

    # Ensure equal aspect ratio for accurate circle representation
    plt.gca().set_aspect('equal', adjustable='box')

    # Set plot limits for better visualization
    plt.xlim([0.3, 1.2])
    plt.ylim([-0.8, 0.8])

    # Save plot to specified directory
    file_name = 'DMD_Eigenvalues_with_Unit_Circle.png'
    plt.savefig(os.path.join(folder_path, file_name))
    plt.close()

    print(f"Eigenvalue plot with unit circle saved in folder: {folder_path}")


if __name__ == "__main__":
    # ===== Experiment setup =====
    np.random.seed(42)         # Set random seed for reproducibility
    S = 5                      # Target rank
    oversample = 25            # Oversampling parameter for randomized algorithms
    n_subspace = 2             # Number of subspace iterations
    p = 20                     # Time-delay embedding order

    # ===== Create result folder =====
    folder_name = create_timestamped_run_folder()

    # ===== Load and preprocess data =====
    data = scipy.io.loadmat('/root/autodl-tmp/Randomized_HOEDMD/JiangnanData.mat')['data']
    data = data[0, 0]
    data = stats.zscore(data, axis=1)  # Normalize data (zero mean, unit variance)
    X_noisy = data

    # ===== Deterministic HOEDMD =====
    start_time = time.time()
    eigenvalues_Deter_HOEDMD, right_eigenvectors_Deter_HOEDMD, left_eigenvectors_hat_Deter_HOEDMD, Phi_x0_Deter_HOEDMD = HOEDMD(X_noisy, p=p, S=S, block=True)
    end_time = time.time()
    print(end_time - start_time)
    sorted_lambda_s_indices_real = np.argsort(eigenvalues_Deter_HOEDMD)[::-1]
    Fmodes_det = np.real(right_eigenvectors_Deter_HOEDMD[:, sorted_lambda_s_indices_real])

    # ===== Randomized HOEDMD =====
    start_time1 = time.time()
    eigenvalues_Randomized_HOEDMD, right_eigenvectors_Randomized_HOEDMD, left_eigenvectors_hat_Randomized_HOEDMD, Phi_x0_Randomized_HOEDMD = Randomized_HOEDMD(
        X_noisy, S, p, oversample=oversample, n_subspace=n_subspace, block=True)
    end_time1 = time.time()
    print(end_time1 - start_time1)
    sorted_lambda_s_indices_pred = np.argsort(eigenvalues_Randomized_HOEDMD)[::-1]
    Fmodes_rand = np.real(right_eigenvectors_Randomized_HOEDMD[:, sorted_lambda_s_indices_pred])

    # ===== Compute standard DMD, randomized DMD, and compressed DMD =====
    Fmodes_det_DMD, l_det_DMD, omega_det = compute_lapack_dmd(X_noisy, rank=S, modes='standard')
    Fmodes_rand_rDMD, l_rand_rdmd, omega_rand = compute_rdmd(X_noisy, rank=S, oversample=oversample, n_subspace=n_subspace)
    _, l_compress_DMD, _ = compute_dmd_compress(X_noisy, rank=S, modes='standard')

    # ===== Save computed modes =====
    sio.savemat('/root/autodl-tmp/Randomized_HOEDMD/Experiment_3/Fmodes_rand.mat', {'Fmodes_rand': Fmodes_rand})
    sio.savemat('/root/autodl-tmp/Randomized_HOEDMD/Experiment_3/Fmodes_det.mat', {'Fmodes_det': Fmodes_det})

    # ===== Plot eigenvalue comparison =====
    plot_and_save_eigenvalues_with_unit_circle(
        eigenvalues_Deter_HOEDMD, eigenvalues_Randomized_HOEDMD, l_det_DMD, l_rand_rdmd, l_compress_DMD, folder_name)
