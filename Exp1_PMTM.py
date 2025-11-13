import sys
import os
# Prepend the parent directory of this file to sys.path so local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from Randomized_HOEDMD import conjugate_transpose, HOEDMD, Randomized_HOEDMD
from evaluation import Evaluate_results,evaluate_rre,evaluate_rre_zhenshide
from datetime import datetime
import time
import tracemalloc



def create_timestamped_run_folder():
    # Create a timestamped output folder based on script name and current time
    # Returns a path like "<script>_runs/YYYYMMDD_HHMMSS"
    """
    Creates a timestamped folder for organizing program runs.
    
    Returns:
        str: Path to the newly created folder in format:
             "{script_name}_runs/YYYYMMDD_HHMMSS/"
    
    Example:
        >>> folder = create_timestamped_run_folder()
        >>> print(folder)
        'my_script_runs/20230815_143022/'
    """
    # Get the current script name without .py extension
    script_name = os.path.basename(__file__).replace('.py', '')
    
    # Generate current timestamp in YYYYMMDD_HHMMSS format
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create folder path combining script name and timestamp
    folder_path = f"{script_name}_runs/{timestamp}"
    
    # Create the directory (including parent 'runs' folder if needed)
    os.makedirs(folder_path, exist_ok=True)
    
    return folder_path


def generate_data():
    # Synthesize a multi-mode linear dynamical system with controllable noise
    # Returns noisy snapshots, true amplitudes, true modes, and true eigenvalues
    """
    Generate synthetic dynamic system data with controlled noise for DMD analysis.
    
    Returns:
        X_noisy: Noisy snapshot matrix (shape M×N)
        a_s: Amplitude coefficients for each mode (shape S,)
        U_zhenshide: True mode shapes (shape M×S)
        lambda_s: Complex eigenvalues for each mode (shape S,)
        para: List of generation parameters [M,R,S,N,noise_level,...]
    """
    # System parameters
    M = 800   # Number of spatial features (measurement points)
    R = 5     # Spatial complexity (rank of spatial modes)
    S = 11    # Spectral complexity (number of dynamic modes)
    N = 1000  # Number of temporal snapshots
    noise_level = 1e-5  # Relative noise level (signal-to-noise ratio)

    # Generate R orthonormal spatial basis vectors (complex-valued)
    Q = np.linalg.qr(np.random.randn(M, R) + 1j * np.random.randn(M, R))[0]

    # Generate S dynamic modes as random linear combinations of spatial basis
    beta = np.random.randn(S, R) + 1j * np.random.randn(S, R)  
    U = np.dot(beta, conjugate_transpose(Q)) 
    U_zhenshide = U.T

    # Generate mode amplitudes (decaying) and angular frequencies
    a_s = np.array([10 ** (-s / 2) for s in range(1, S + 1)])
    omega_s = np.array([10 * s for s in range(1, S + 1)])

    # Initialize clean data matrix
    X = np.zeros((M, N), dtype=complex)
    t = np.linspace(0, 1, N)  # Time vector (normalized to [0,1])

    # Create eigenvalues on (or near) the unit circle for stability
    lambda_s = np.exp(1j * omega_s / (N - 1))

    # Build clean signal as a superposition of modal evolutions
    X_noisy = np.zeros_like(X)
    for n in range(N):
        for s in range(S):
            X_noisy[:, n] += a_s[s] * U[s, :] * lambda_s[s] ** (n - 1)

    # ===== Noise generation with energy control =====
    # Compute total signal energy
    norms = np.linalg.norm(X_noisy, axis=0)
    E_X = np.sum(norms**2)
    
    # Target noise energy from relative noise level
    E_Z = E_X * noise_level**2
    
    # Generate i.i.d. Gaussian noise and scale to desired energy
    Z = np.random.normal(size=X_noisy.shape)  
    Z_norms = np.linalg.norm(Z, axis=0)
    scaling_factor = np.sqrt(E_Z / np.sum(Z_norms**2))
    Z_scaled = Z * scaling_factor

    # Add scaled noise to the clean signal
    X_noisy += Z_scaled
    
    # Recompute energies and print SNR for verification
    E_X = np.sum(np.linalg.norm(X_noisy, axis=0)**2)
    E_Z = np.sum(np.linalg.norm(Z_scaled, axis=0)**2)
    snr_db = 10 * np.log10(E_X / E_Z)
    
    print(f"Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")

    return X_noisy,a_s,U_zhenshide,lambda_s





if __name__ == "__main__":

    # Global configuration: randomness, model sizes, and algorithmic knobs
    # S: number of modes; oversample/n_subspace: rHOEDMD sketching parameters
    # orderp: list of time-delay embedding orders to sweep
    # loading
    np.random.seed(42)  
    S = 11
    oversample=10
    n_subspace=2
    orderp = [2,4,8,16,32,64]

    # Prepare a unique output directory for logs and CSVs
    # folder_name creating
    folder_name = create_timestamped_run_folder()

    

    
    # Global accumulators across all p for each metric
    rrmse_modes_Deter_HOEDMD = []
    rrmse_modes_Randomized_HOEDMD = []
    rrmse_poles_Deter_HOEDMD = []
    rrmse_poles_Randomized_HOEDMD = []
    rrs_Deter_HOEDMD = []
    rrs_Randomized_HOEDMD = []
    time_Deter_HOEDMD = []
    time_Randomized_HOEDMD = []
    memory_Deter_HOEDMD = []
    memory_Randomized_HOEDMD = []

    for p in orderp:
        # Per-p accumulators to store 100-trial statistics
        rrmse_modes_Deter_HOEDMD_eachp = []
        rrmse_modes_Randomized_HOEDMD_eachp = []
        rrmse_poles_Deter_HOEDMD_eachp = []
        rrmse_poles_Randomized_HOEDMD_eachp = []
        rrs_Deter_HOEDMD_eachp = []
        rrs_Randomized_HOEDMD_eachp = []
        time_Deter_HOEDMD_eachp = []
        time_Randomized_HOEDMD_eachp = []
        memory_Deter_HOEDMD_eachp = []
        memory_Randomized_HOEDMD_eachp = []

        for _ in range(100):
            # Status message for current trial
            print('p=', p, 'experiment', _, 'completed')

            # Generate a fresh synthetic dataset per trial
            # Returns noisy snapshots and the ground-truth modal quantities
            X_noisy,a_s,U_zhenshide,lambda_s = generate_data()

            # ===== Deterministic HOEDMD timing + peak memory =====
            tracemalloc.start()                      # begin memory tracing
            start_time1 = time.time()
            eigenvalues_Deter_HOEDMD,right_eigenvectors_Deter_HOEDMD,left_eigenvectors_hat_Deter_HOEDMD,Phi_x0_Deter_HOEDMD = HOEDMD(X_noisy,p=p,S=S)
            end_time1 = time.time()
            time_Deter_HOEDMD_eachp.append(end_time1 - start_time1)
            _, peak1 = tracemalloc.get_traced_memory()  # peak bytes during trace
            tracemalloc.stop()                          # end memory tracing
            memory_Deter_HOEDMD_eachp.append(peak1 / (1024 * 1024))  # bytes -> MB

            # Reset tracer state before next measurement
            tracemalloc.clear_traces()
            
            # ===== Randomized HOEDMD timing + peak memory =====
            tracemalloc.start()
            start_time2 = time.time()
            eigenvalues_Randomized_HOEDMD,right_eigenvectors_Randomized_HOEDMD,left_eigenvectors_hat_Randomized_HOEDMD,Phi_x0_Randomized_HOEDMD = Randomized_HOEDMD(X_noisy,S,p,oversample=oversample, n_subspace=n_subspace)
            end_time2 = time.time()
            time_Randomized_HOEDMD_eachp.append(end_time2 - start_time2)    
            _, peak2 = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_Randomized_HOEDMD_eachp.append(peak2 / (1024 * 1024))

            # ===== Accuracy/evaluation metrics =====
            # Compare predicted eigenpairs/modes to ground truth for both methods
            poles_error_Deter_HOEDMD,modes_error_Deter_HOEDMD = Evaluate_results(a_s,U_zhenshide,lambda_s,eigenvalues_Deter_HOEDMD,right_eigenvectors_Deter_HOEDMD,left_eigenvectors_hat_Deter_HOEDMD,Phi_x0_Deter_HOEDMD)
            poles_error_Randomized_HOEDMD,modes_erro_Randomized_HOEDMD = Evaluate_results(a_s,U_zhenshide,lambda_s,eigenvalues_Randomized_HOEDMD,right_eigenvectors_Randomized_HOEDMD,left_eigenvectors_hat_Randomized_HOEDMD,Phi_x0_Randomized_HOEDMD)
            
            # Relative reconstruction error of the time series under (H)OEDMD models
            rre_error_Deter_HOEDMD = evaluate_rre(X_noisy,eigenvalues_Deter_HOEDMD,right_eigenvectors_Deter_HOEDMD,left_eigenvectors_hat_Deter_HOEDMD,Phi_x0_Deter_HOEDMD,HOEDMD_flag=True)
            rre_error_Randomized_HOEDMD = evaluate_rre(X_noisy,eigenvalues_Randomized_HOEDMD,right_eigenvectors_Randomized_HOEDMD,left_eigenvectors_hat_Randomized_HOEDMD,Phi_x0_Randomized_HOEDMD,HOEDMD_flag=True)

            # Record trial metrics for this p
            # Record_each_p
            rrmse_modes_Deter_HOEDMD_eachp.append(modes_error_Deter_HOEDMD)
            rrmse_modes_Randomized_HOEDMD_eachp.append(modes_erro_Randomized_HOEDMD)
            rrmse_poles_Deter_HOEDMD_eachp.append(poles_error_Deter_HOEDMD)
            rrmse_poles_Randomized_HOEDMD_eachp.append(poles_error_Randomized_HOEDMD)
            rrs_Deter_HOEDMD_eachp.append(rre_error_Deter_HOEDMD)
            rrs_Randomized_HOEDMD_eachp.append(rre_error_Randomized_HOEDMD)
        
        # Append per-p arrays to global lists
        # Record
        rrmse_modes_Deter_HOEDMD.append(rrmse_modes_Deter_HOEDMD_eachp)
        rrmse_modes_Randomized_HOEDMD.append(rrmse_modes_Randomized_HOEDMD_eachp)
        rrmse_poles_Deter_HOEDMD.append(rrmse_poles_Deter_HOEDMD_eachp)
        rrmse_poles_Randomized_HOEDMD.append(rrmse_poles_Randomized_HOEDMD_eachp)
        rrs_Deter_HOEDMD.append(rrs_Deter_HOEDMD_eachp)
        rrs_Randomized_HOEDMD.append(rrs_Randomized_HOEDMD_eachp)
        time_Deter_HOEDMD.append(time_Deter_HOEDMD_eachp)
        time_Randomized_HOEDMD.append(time_Randomized_HOEDMD_eachp)
        memory_Deter_HOEDMD.append(memory_Deter_HOEDMD_eachp)
        memory_Randomized_HOEDMD.append(memory_Randomized_HOEDMD_eachp)

    # ===== Serialize metrics to CSV files in the run folder =====
    Time_df = {
    'STLS-based HOEDMD': time_Deter_HOEDMD,
    'Randomized STLS-based HOEDMD': time_Randomized_HOEDMD,
    }
    Time_df = pd.DataFrame.from_dict(Time_df)
    time_outputpath = os.path.join(folder_name, 'time.csv')
    Time_df.to_csv(time_outputpath, index=False)

    Memory_df = {
    'STLS-based HOEDMD': memory_Deter_HOEDMD,
    'Randomized STLS-based HOEDMD': memory_Randomized_HOEDMD,
    }
    Memory_df = pd.DataFrame.from_dict(Memory_df)
    memory_outputpath = os.path.join(folder_name, 'memory.csv')
    Memory_df.to_csv(memory_outputpath, index=False)
    
    Rrs_df = {
    'STLS-based HOEDMD': rrs_Deter_HOEDMD,
    'Randomized STLS-based HOEDMD': rrs_Randomized_HOEDMD,
    }
    Rrs_df = pd.DataFrame.from_dict(Rrs_df)
    rrs_outputpath = os.path.join(folder_name, 'rrs.csv')
    Rrs_df.to_csv(rrs_outputpath, index=False)

    Poles_df = {
    'STLS-based HOEDMD': rrmse_poles_Deter_HOEDMD,
    'Randomized STLS-based HOEDMD': rrmse_poles_Randomized_HOEDMD,
    }
    Poles_df = pd.DataFrame.from_dict(Poles_df)
    poles_outputpath = os.path.join(folder_name, 'poles.csv')
    Poles_df.to_csv(poles_outputpath, index=False)

    Modes_df = {
    'STLS-based HOEDMD': rrmse_modes_Deter_HOEDMD,
    'Randomized STLS-based HOEDMD': rrmse_modes_Randomized_HOEDMD,
    }
    Modes_df = pd.DataFrame.from_dict(Modes_df)
    mdoes_outputpath = os.path.join(folder_name, 'modes.csv')
    Modes_df.to_csv(mdoes_outputpath, index=False)
