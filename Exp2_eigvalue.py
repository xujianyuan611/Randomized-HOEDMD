import sys
import os
# Add the parent directory of this file to sys.path so local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from Randomized_HOEDMD import conjugate_transpose, HOEDMD, Randomized_HOEDMD
from evaluation import Evaluate_results,evaluate_rre,Evaluate_results_deter_pred,calculate_rrmse_poles,calculate_rrmse_modes
from datetime import datetime
import time
import scipy.io
from rDMD import compute_dmd,compute_rdmd
from lapack_DMD import compute_lapack_dmd
from compress_DMD import compute_dmd_compress
import matplotlib.pyplot as plt

def save_modes(modes, method_name, grid_shape, folder_path):
    """
    Save contour plots of the first 6 DMD modes.

    Parameters
    ----------
    modes : ndarray
        Matrix whose columns are spatial modes (flattened).
    method_name : str
        Name used in the output filename to indicate the method.
    grid_shape : tuple
        Shape to which each mode column is reshaped for plotting.
    folder_path : str
        Directory where the image file is saved.
    """
    # Configure subplots; using a negative hspace to reduce vertical spacing
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), 
                            gridspec_kw={'wspace': 0.1, 'hspace': -0.3})  # Reduce vertical spacing
    
    axes = axes.ravel()

    modes_to_plot = 6  # Number of modes to visualize
    
    for i in range(modes_to_plot):
        mode = modes[:, i].reshape(grid_shape)
        ax = axes[i]
        c = ax.contourf(mode.T, levels=50, cmap='viridis')
        ax.set_aspect('equal')
        
        # Hide tick marks
        ax.set_xticks([])  # Hide x-axis ticks
        ax.set_yticks([])  # Hide y-axis ticks
        
        # Optionally hide axis frame
        ax.axis('off')
        
        # Add a filled black circle at (50, 100) with radius 25 as a visual marker
        circle = plt.Circle((50, 100), 25, color='black', fill=True)
        ax.add_patch(circle)
    
    # Save figure to the specified folder
    file_name = f'{method_name}_DMD_Modes.png'
    plt.savefig(os.path.join(folder_path, file_name), dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def generate_gaussian_noise(data, noise_level=0.1):
    """Generate Gaussian noise with specified relative level."""
    return noise_level * np.std(data) * np.random.randn(*data.shape)

def plot_dual_eigenvalues_with_unit_circle(noiseless_data, noisy_data, folder_path):
    """
    Create a two-panel figure comparing eigenvalues for noiseless vs. noisy cases.
    A common legend is placed at the top.

    Parameters
    ----------
    noiseless_data : tuple
        (l_det, l_rand, l_det1, l_rand1, l_compress) for noiseless data.
    noisy_data : tuple
        (l_det, l_rand, l_det1, l_rand1, l_compress) for noisy data.
    folder_path : str
        Directory to save the figure.
    """
    # Unpack data
    l_det, l_rand, l_det1, l_rand1, l_compress = noiseless_data
    l_det_n, l_rand_n, l_det1_n, l_rand1_n, l_compress_n = noisy_data
    
    # Create a 1x2 figure (noiseless on left, noisy on right)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Optional overall title is commented out
    # fig.suptitle('DMD Eigenvalues Comparison: Noiseless vs. Noisy Data', fontsize=16)
    
    # Left subplot: noiseless case
    plot_eigenvalues_on_axis(axs[0], l_det, l_rand, l_det1, l_rand1, l_compress)
    axs[0].set_title('(a) Noiseless Data', fontsize=15)
    
    # Right subplot: noisy case
    plot_eigenvalues_on_axis(axs[1], l_det_n, l_rand_n, l_det1_n, l_rand1_n, l_compress_n)
    axs[1].set_title('(b) Noisy Data', fontsize=15)
    
    # Create a unified legend across subplots, placed at the top center
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
               ncol=5, fontsize=15, frameon=True, 
               bbox_to_anchor=(0.5, 1.0), 
               bbox_transform=fig.transFigure)
    
    # Layout adjustments to leave space for the top legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save the comparison figure
    file_name = 'DMD_Eigenvalues_Comparison.png'
    plt.savefig(os.path.join(folder_path, file_name), dpi=300)
    plt.close()
    
    print(f"Comparison plot saved in folder: {folder_path}")

def plot_eigenvalues_on_axis(ax, l_det, l_rand, l_det1, l_rand1, l_compress):
    """Plot eigenvalues of multiple methods on a given axis, including the unit circle."""
    # Extract real/imag parts for each method's eigenvalues
    eigvals_det_real = l_det.real
    eigvals_det_imag = l_det.imag
    eigvals_rand_real = l_rand.real
    eigvals_rand_imag = l_rand.imag
    eigvals_det_real1 = l_det1.real
    eigvals_det_imag1 = l_det1.imag
    eigvals_rand_real1 = l_rand1.real
    eigvals_rand_imag1 = l_rand1.imag
    eigvals_compress_real1 = l_compress.real
    eigvals_compress_imag1 = l_compress.imag

    # Scatter plots for each method with distinct markers/colors
    det_scatter = ax.scatter(eigvals_det_real, eigvals_det_imag, color='b', 
                            label='Det HOEDMD', 
                            alpha=0.7, marker='s', s=100)
    rand_scatter = ax.scatter(eigvals_rand_real, eigvals_rand_imag, color='r', 
                             label='Rand HOEDMD', 
                             alpha=0.7, marker='*', s=150)
    det_scatter1 = ax.scatter(eigvals_det_real1, eigvals_det_imag1, color='yellow', 
                             label='Det DMD', 
                             alpha=0.7, marker='^', s=140)
    rand_scatter1 = ax.scatter(eigvals_rand_real1, eigvals_rand_imag1, color='g', 
                              label='Rand DMD', 
                              alpha=0.7, marker='o', s=60)
    compress_scatter1 = ax.scatter(eigvals_compress_real1, eigvals_compress_imag1, color='#DDA0DD', 
                              label='Comp DMD', 
                              alpha=0.7, marker='d', s=50)

    # Draw the unit circle (radius 1) for stability reference
    unit_circle = plt.Circle((0, 0), 1, color='black', fill=False, 
                            linestyle='--', linewidth=1.5)
    ax.add_artist(unit_circle)

    # Axis labels
    ax.set_xlabel('Real Part', fontsize=15)
    ax.set_ylabel('Imaginary Part', fontsize=15)

    # Add grid and axes lines for reference
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

    # Font size for tick labels
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    
    # Equal aspect ratio to keep the unit circle circular; set plot limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])

def create_timestamped_run_folder():
    """Create a folder named with a timestamp (run_YYYYMMDD_HHMMSS) in the current directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"run_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

# Assume compute_dmd and compute_rdmd functions are already imported/defined above

if __name__ == "__main__":
    # ==== Initialization ====
    np.random.seed(4211)   # Set seed for reproducible noise/randomness
    S = 15                 # Target rank (number of modes)
    oversample = 25        # Oversampling parameter for randomized algorithms
    n_subspace = 8         # Number of subspace iterations in randomized sketching
    p = 20                 # Time-delay embedding order for HOEDMD

    # Create an output directory for this run
    folder_name = create_timestamped_run_folder()

    # Load data from MAT file
    data = scipy.io.loadmat('/root/autodl-tmp/Randomized_HOEDMD/CYLINDER_ALL.mat')
    X_original = data['VORTALL']  # Original snapshot matrix

    # ===== Case 1: Noiseless data =====
    X_noiseless = X_original.copy()
    
    # Deterministic HOEDMD on noiseless data
    start_time = time.time()
    eigenvalues_Deter_HOEDMD, _, _, _ = HOEDMD(X_noiseless, p=p, S=S, block=True)
    end_time = time.time()
    print(f"Noiseless HOEDMD time: {end_time - start_time:.2f} seconds")
    
    # Randomized HOEDMD on noiseless data
    start_time1 = time.time()
    eigenvalues_Randomized_HOEDMD, _, _, _ = Randomized_HOEDMD(
        X_noiseless, S, p, oversample=oversample, n_subspace=n_subspace, block=True)
    end_time1 = time.time()
    print(f"Noiseless Randomized HOEDMD time: {end_time1 - start_time1:.2f} seconds")
    
    # Standard DMD (LAPACK/NumPy-based) and compressed DMD on noiseless data
    _, l_det_DMD, _ = compute_lapack_dmd(X_noiseless, rank=S, modes='standard')
    _, l_compress_DMD, _ = compute_dmd_compress(X_noiseless, rank=S, modes='standard')

    # Randomized DMD (QB-based) on noiseless data
    _, l_rand_rdmd, _ = compute_rdmd(X_noiseless, rank=S, oversample=oversample, n_subspace=n_subspace)
    
    # Bundle noiseless results for plotting
    noiseless_results = (
        eigenvalues_Deter_HOEDMD, 
        eigenvalues_Randomized_HOEDMD,
        l_det_DMD,
        l_rand_rdmd,
        l_compress_DMD
    )

    # ===== Case 2: Noisy data =====
    X_noisy = X_original.copy()
    
    # Add Gaussian noise with target SNR of 10 dB
    SNR_dB = 10
    signal_power = np.mean(X_noisy**2)
    SNR_linear = 10**(SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.random.normal(0, np.sqrt(noise_power), X_noisy.shape)
    X_noisy += noise
    
    # Deterministic HOEDMD on noisy data
    start_time = time.time()
    eigenvalues_Deter_HOEDMD_n, right_eigenvectors_Deter_HOEDMD_n, _, _ = HOEDMD(X_noisy, p=p, S=S, block=True)
    end_time = time.time()
    print(f"Noisy HOEDMD time: {end_time - start_time:.2f} seconds")
    
    # Randomized HOEDMD on noisy data
    start_time1 = time.time()
    eigenvalues_Randomized_HOEDMD_n, right_eigenvectors_Randomized_HOEDMD_n, _, _ = Randomized_HOEDMD(
        X_noisy, S, p, oversample=oversample, n_subspace=n_subspace, block=True)
    end_time1 = time.time()
    print(f"Noisy Randomized HOEDMD time: {end_time1 - start_time1:.2f} seconds")
    
    # Standard and compressed DMD on noisy data
    _, l_det_DMD_n, _ = compute_lapack_dmd(X_noisy, rank=S, modes='standard')
    _, l_compress_DMD_n, _ = compute_dmd_compress(X_noisy, rank=S, modes='standard')

    # Randomized DMD on noisy data
    _, l_rand_rdmd_n, _ = compute_rdmd(X_noisy, rank=S, oversample=oversample, n_subspace=n_subspace)
    
    # Bundle noisy results for plotting
    noisy_results = (
        eigenvalues_Deter_HOEDMD_n, 
        eigenvalues_Randomized_HOEDMD_n,
        l_det_DMD_n,
        l_rand_rdmd_n,
        l_compress_DMD_n
    )

    # Grid shape for reshaping flattened modes (domain-specific)
    grid_shape = (449, 199)

    # Sort and take real parts of HOEDMD modes for deterministic case
    sorted_lambda_s_indices_real = np.argsort(eigenvalues_Deter_HOEDMD_n)[::-1]
    Fmodes_det = np.real(right_eigenvectors_Deter_HOEDMD_n[:,sorted_lambda_s_indices_real])

    # Sort and take (negated) real parts of randomized HOEDMD modes (as in original script)
    sorted_lambda_s_indices_pred = np.argsort(eigenvalues_Randomized_HOEDMD_n)[::-1]
    Fmodes_rand = -np.real(right_eigenvectors_Randomized_HOEDMD_n[:,sorted_lambda_s_indices_pred])

    # Save the first six deterministic DMD modes
    save_modes(Fmodes_det, "Deterministic", grid_shape, folder_name)

    # Save the first six randomized DMD modes
    save_modes(Fmodes_rand, "Randomized", grid_shape, folder_name)

    # Plot side-by-side eigenvalue comparisons for noiseless vs noisy cases
    plot_dual_eigenvalues_with_unit_circle(noiseless_results, noisy_results, folder_name)
