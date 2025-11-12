import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Randomized_HOEDMD import conjugate_transpose, HOEDMD, Randomized_HOEDMD
from evaluation import Evaluate_results, evaluate_rre, Evaluate_results_deter_pred, calculate_rrmse_poles, calculate_rrmse_modes

from scipy import stats
import scipy
import scipy.io as sio
import time
import numpy as np


if __name__ == "__main__":
    # ===== Set up experiment parameters =====
    np.random.seed(42)           # Fix random seed for reproducibility
    S = 8                        # Target rank or number of modes
    oversample = 15              # Oversampling parameter for randomized HOEDMD
    n_subspace = 5               # Number of subspace iterations for randomized SVD

    # Iterate through different time-delay embedding dimensions p
    for p in [1, 2, 4, 8, 16, 32]:
        # Initialize lists for timing and error results across datasets
        data1_time_det = []      # Deterministic HOEDMD runtime (Dataset 1)
        data1_time_rand = []     # Randomized HOEDMD runtime (Dataset 1)
        data1_modes_rand = []    # Randomized HOEDMD mode errors (Dataset 1)
        data1_poles_rand = []    # Randomized HOEDMD pole errors (Dataset 1)
        data2_time_det = []      # Deterministic HOEDMD runtime (Dataset 2)
        data2_time_rand = []     # Randomized HOEDMD runtime (Dataset 2)
        data2_modes_rand = []    # Randomized HOEDMD mode errors (Dataset 2)
        data2_poles_rand = []    # Randomized HOEDMD pole errors (Dataset 2)
        data3_time_det = []      # Deterministic HOEDMD runtime (Dataset 3)
        data3_time_rand = []     # Randomized HOEDMD runtime (Dataset 3)
        data3_modes_rand = []    # Randomized HOEDMD mode errors (Dataset 3)
        data3_poles_rand = []    # Randomized HOEDMD pole errors (Dataset 3)

        # Repeat each experiment 100 times for statistical averaging
        for _ in range(100):
            print('p = ', p, 'Experiment', _)

            # ===== Dataset 1: JiangnanData.mat =====
            data1 = scipy.io.loadmat('/root/autodl-tmp/Randomized_HOEDMD/JiangnanData.mat')['data'][0, 0]
            data1 = stats.zscore(data1, axis=1)  # Normalize data (zero mean, unit variance)
            X_noisy = data1

            # ---- Deterministic HOEDMD ----
            start_time = time.time()
            eigenvalues_Deter_HOEDMD, right_eigenvectors_Deter_HOEDMD, left_eigenvectors_hat_Deter_HOEDMD, Phi_x0_Deter_HOEDMD = HOEDMD(X_noisy, p=p, S=S)
            end_time = time.time()
            data1_time_det.append(end_time - start_time)

            # ---- Randomized HOEDMD ----
            start_time1 = time.time()
            eigenvalues_Randomized_HOEDMD, right_eigenvectors_Randomized_HOEDMD, left_eigenvectors_hat_Randomized_HOEDMD, Phi_x0_Randomized_HOEDMD = Randomized_HOEDMD(
                X_noisy, S, p, oversample=oversample, n_subspace=n_subspace)
            end_time1 = time.time()
            data1_time_rand.append(end_time1 - start_time1)
           
            # ---- Evaluate reconstruction errors ----
            poles_error_Randomized_HOEDMD, modes_error_Randomized_HOEDMD = Evaluate_results_deter_pred(
                eigenvalues_Deter_HOEDMD, right_eigenvectors_Deter_HOEDMD, left_eigenvectors_hat_Deter_HOEDMD, Phi_x0_Deter_HOEDMD,
                eigenvalues_Randomized_HOEDMD, right_eigenvectors_Randomized_HOEDMD, left_eigenvectors_hat_Randomized_HOEDMD, Phi_x0_Randomized_HOEDMD)
            
            data1_modes_rand.append(modes_error_Randomized_HOEDMD)
            data1_poles_rand.append(poles_error_Randomized_HOEDMD)
            
            
            # ===== Dataset 2: QianchengyuanYATC_157942.mat =====
            data2 = sio.loadmat('/root/autodl-tmp/Randomized_HOEDMD/QianchengyuanYATC_157942.mat')['TC']
            X_noisy = data2

            # ---- Deterministic HOEDMD ----
            start_time2 = time.time()
            eigenvalues_Deter_HOEDMD, right_eigenvectors_Deter_HOEDMD, left_eigenvectors_hat_Deter_HOEDMD, Phi_x0_Deter_HOEDMD = HOEDMD(X_noisy, p=p, S=S)
            end_time2 = time.time()
            data2_time_det.append(end_time2 - start_time2)

            # ---- Randomized HOEDMD ----
            start_time3 = time.time()
            eigenvalues_Randomized_HOEDMD, right_eigenvectors_Randomized_HOEDMD, left_eigenvectors_hat_Randomized_HOEDMD, Phi_x0_Randomized_HOEDMD = Randomized_HOEDMD(
                X_noisy, S, p, oversample=oversample, n_subspace=n_subspace)
            end_time3 = time.time()
            data2_time_rand.append(end_time3 - start_time3)

            # ---- Evaluate reconstruction errors ----
            poles_error_Randomized_HOEDMD, modes_error_Randomized_HOEDMD = Evaluate_results_deter_pred(
                eigenvalues_Deter_HOEDMD, right_eigenvectors_Deter_HOEDMD, left_eigenvectors_hat_Deter_HOEDMD, Phi_x0_Deter_HOEDMD,
                eigenvalues_Randomized_HOEDMD, right_eigenvectors_Randomized_HOEDMD, left_eigenvectors_hat_Randomized_HOEDMD, Phi_x0_Randomized_HOEDMD)
            
            data2_modes_rand.append(modes_error_Randomized_HOEDMD)
            data2_poles_rand.append(poles_error_Randomized_HOEDMD)



            # ===== Dataset 3: WeitongFmri662857.mat =====
            file_path = "/root/autodl-tmp/Randomized_HOEDMD/WeitongFmri662857.mat"
            data3 = sio.loadmat(file_path)['C']
            X_noisy = data3

            # ---- Deterministic HOEDMD ----
            start_time4 = time.time()
            eigenvalues_Deter_HOEDMD, right_eigenvectors_Deter_HOEDMD, left_eigenvectors_hat_Deter_HOEDMD, Phi_x0_Deter_HOEDMD = HOEDMD(X_noisy, p=p, S=S)
            end_time4 = time.time()
            data3_time_det.append(end_time4 - start_time4)

            # ---- Randomized HOEDMD ----
            start_time5 = time.time()
            eigenvalues_Randomized_HOEDMD, right_eigenvectors_Randomized_HOEDMD, left_eigenvectors_hat_Randomized_HOEDMD, Phi_x0_Randomized_HOEDMD = Randomized_HOEDMD(
                X_noisy, S, p, oversample=oversample, n_subspace=n_subspace)
            end_time5 = time.time()
            data3_time_rand.append(end_time5 - start_time5)

            # ---- Error evaluation ----
            poles_error_Randomized_HOEDMD,modes_error_Randomized_HOEDMD = Evaluate_results_deter_pred(eigenvalues_Deter_HOEDMD,right_eigenvectors_Deter_HOEDMD,left_eigenvectors_hat_Deter_HOEDMD,Phi_x0_Deter_HOEDMD,eigenvalues_Randomized_HOEDMD,right_eigenvectors_Randomized_HOEDMD,left_eigenvectors_hat_Randomized_HOEDMD,Phi_x0_Randomized_HOEDMD)
            data3_poles_rand.append(poles_error_Randomized_HOEDMD)
            data3_modes_rand.append(modes_error_Randomized_HOEDMD)

        # ===== Print average and standard deviation of results =====
        print('data1,time,det: mean:', np.mean(data1_time_det), 'std:', np.std(data1_time_det))
        print('data1,time,rand: mean:', np.mean(data1_time_rand), 'std:', np.std(data1_time_rand))
        print('data1,modes,rand: mean:', np.mean(data1_modes_rand), 'std:', np.std(data1_modes_rand))
        print('data1,poles,rand: mean:', np.mean(data1_poles_rand), 'std:', np.std(data1_poles_rand))

        print('data2,time,det: mean:', np.mean(data2_time_det), 'std:', np.std(data2_time_det))
        print('data2,time,rand: mean:', np.mean(data2_time_rand), 'std:', np.std(data2_time_rand))
        print('data2,modes,rand: mean:', np.mean(data2_modes_rand), 'std:', np.std(data2_modes_rand))
        print('data2,poles,rand: mean:', np.mean(data2_poles_rand), 'std:', np.std(data2_poles_rand))

        print('data3,time,det: mean:', np.mean(data3_time_det), 'std:', np.std(data3_time_det))
        print('data3,time,rand: mean:', np.mean(data3_time_rand), 'std:', np.std(data3_time_rand))
        print('data3,modes,rand: mean:', np.mean(data3_modes_rand), 'std:', np.std(data3_modes_rand))
        print('data3,poles,rand: mean:', np.mean(data3_poles_rand), 'std:', np.std(data3_poles_rand))
