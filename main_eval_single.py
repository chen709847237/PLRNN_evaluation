import os

from scipy.io import loadmat, savemat
import numpy as np
import torch as tc
import pandas as pd
from matplotlib import pyplot as plt
import matlab.engine
from klx_gmm import calc_kl_from_data
from pse import power_spectrum_error_per_dim
import random


def printf(x):
    if PRINT:
        print(x)

def setup_seed(seed):
    tc.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #tc.backends.cudnn.deterministic = True

#############################################################

def step_forward(z, A, W, h):
    z_activated = tc.relu(z)
    z = A * z + z_activated @ W.t() + h

    return z

def output_mapping(z, H, M, T, B_est, M_est, R):

    # B_est             [19, 8]
    # M_est = net.J     [19, 8]
    # R                 [132, 8]
    # H                 [1056, 1056]

    z_1d = z.reshape(1, tc.numel(z)).t()                # [132, 8] ---> [1056, 1]
    hZ = tc.mm(H, z_1d)                                 # [1056, 1] hZ_fr = H * Z_fr_1d'
    hZ = hZ.reshape(T, M).t()                           # [8, 132]  hZ_fr = reshape(hZ_fr,M,T)
    x = tc.add(tc.mm(B_est, hZ), tc.mm(M_est, R.t()))   # [19, 132] X_fr_pre = B_est * hZ_fr + M_est *R';

    return x.t()

def get_ahead_pred_obs(data_true, z, A, W, h, B_est, M_est, R, TR, n_steps, matlab_eng):

    data_true = tc.tensor(data_true).t()    # [132, 19]
    z = tc.tensor(z).t()                    # [132, 8] Ezi
    A = tc.diag(tc.tensor(A))               # [8]
    W = tc.tensor(W)                        # [8, 8]
    h = tc.tensor(h).squeeze(1)             # [8]

    B_est = tc.tensor(B_est)                # [19, 8]
    M_est = tc.tensor(M_est)                # [19, 8]
    R = tc.tensor(R)                        # [132, 8]

    # Data Dims
    T, q = data_true.size()                 # T=132 q=19
    _, M = z.size()                         # M=8

    # true data when n_steps = 5
    time_steps = T - n_steps                # time_steps=127

    z = z[:-n_steps, :]                     # Ezi 132x8 --> 127x8
    R = R[:-n_steps, :]                     # R 132x8 --> 127x8

    # Based on length of time point, generate matrices with different sizes
    H = tc.tensor(matlab_eng.hrf_matrix(time_steps, M, float(TR)), dtype=tc.float64)

    X_pred = tc.empty((n_steps, time_steps, q))

    # n-step forward prediction
    for step in range(n_steps):

        z = step_forward(z, A, W, h)                                 # [127, 8]
        x = output_mapping(z, H, M, time_steps, B_est, M_est, R)     # [127, 19]
        X_pred[step] = x                                             # 5x127x19

    return X_pred

def construct_ground_truth(data_true, n_steps):

    data_true = tc.tensor(data_true).t()    # [132, 19]

    ####### TEST ONLY #######
    #plt.plot(data_true[:, -1], color='k')
    #plt.plot(data_true[:, -1], '.', color='k')
    ####### TEST ONLY #######

    T, q = data_true.size()                 # T=132 q=19

    time_steps = T - n_steps                # time_steps=127
    X_true = tc.empty((n_steps, time_steps, q))

    # step = 1~5
    for step in range(1, n_steps + 1):

        X_true[step - 1] = data_true[step:time_steps+step]

    return X_true

def mse_evaluator(data_true, Ezi, A, W, h, B_est, M_est, R, TR, matlab_eng, n_steps=20):

    printf('@@@ MSE RESULTS @@@')

    print('n_steps:', n_steps)
    #get_ahead_pred_obs
    x_pred = get_ahead_pred_obs(data_true, Ezi, A, W, h, B_est, M_est, R, TR, n_steps, matlab_eng)
    x_true = construct_ground_truth(data_true, n_steps)

    mse_results = tc.pow(x_pred - x_true, 2).mean([1, 2]).numpy()

    for step in [1, 5, 10, 20]:
        printf('# MSE-{} = {}'.format(step, mse_results[step - 1]))

    return mse_results

#############################################################

def factor_checking(target_num):
    quotien = target_num
    num_checking = 1
    factors = []
    while num_checking < quotien:
        if target_num % num_checking == 0:
            quotien = target_num // num_checking
            factors.append(num_checking)
            factors.append(quotien)
        num_checking += 1

    return factors

def free_run(z, T, M, A, W, h):

    Z = tc.empty((T, 1, M), dtype=tc.float64)
    Z[0] = z

    # FREE RUN
    for t in range(1, T):
        Z[t] = step_forward(Z[t-1], A, W, h)

    return Z.squeeze_(1)

def generate_free_trajectory(z, T_sub, M, A, W, h):

    # optionally predict an initial z0 of shape (1, dz)
    z0 = z[0, :].unsqueeze(0)

    # latent traj is T x dz
    latent_traj = free_run(z0, T_sub, M, A, W, h)

    return latent_traj

def klx_evaluator(data_true, Ezi, A, W, h, B_est, M_est, R, TR, H):

    printf('@@@ KLX RESULTS @@@')

    data_true = tc.tensor(data_true).t()        # [132, 19]
    z = tc.tensor(Ezi).t()                      # [132, 8] Ezi
    A = tc.diag(tc.tensor(A))                   # [8]
    W = tc.tensor(W)                            # [8, 8]
    h = tc.tensor(h).squeeze(1)                 # [8]

    B_est = tc.tensor(B_est)                    # [19, 8]
    M_est = tc.tensor(M_est)                    # [19, 8]
    R = tc.tensor(R)                            # [132, 8]

    T, q = data_true.size()                     # T=132 q=19
    _, M = z.size()                             # M=8

    # Init Conditions
    print('INITCONDS candidates: ', factor_checking(data_true.shape[0]))
    # global INITCONDS
    print('INITCONDS = ', INITCONDS)         # INITCONDS = 11

    z_reshaped = z.reshape(INITCONDS, -1, M)                # z_reshaped  ==> [11, 12, 8]
    T_sub = z_reshaped.shape[1]                             # T_sub = 12

    trajs = []
    for traj in z_reshaped:

        Z = generate_free_trajectory(traj, T_sub, M, A, W, h)
        trajs.append(Z)

        ####### TEST ONLY #######
        # break
        ####### TEST ONLY #######

    # [132, 8]
    z_gen = tc.stack(trajs).reshape(T, -1)

    # Generate hrf matrix with different size
    if INITCONDS == 1:
        H = tc.tensor(H, dtype=tc.float64)
    else:
        H = tc.tensor(matlab_eng.hrf_matrix(T, M, float(TR)), dtype=tc.float64)

    # map to X [132, 19]
    data_gen = output_mapping(z_gen, H, M, T, B_est, M_est, R)

    # compute D_stsp
    print("Computing KLx-GMM...")
    klx_value = calc_kl_from_data(data_gen.reshape(-1, q), data_true)

    printf('# KLx = {}'.format(klx_value.item()))

    return [np.array(klx_value.numpy())]

#############################################################

def pse_evaluator(data_true, Ezi, A, W, h, B_est, M_est, R, H):     # TR

    printf('@@@ PSE RESULTS @@@')

    data_true = tc.tensor(data_true).t()        # [132, 19]
    z = tc.tensor(Ezi).t()                      # [132, 8] Ezi
    A = tc.diag(tc.tensor(A))                   # [8] 对角矩阵A 只取对角元素
    W = tc.tensor(W)                            # [8, 8]
    h = tc.tensor(h).squeeze(1)                 # [8]

    B_est = tc.tensor(B_est)                    # [19, 8]
    M_est = tc.tensor(M_est)                    # [19, 8]
    R = tc.tensor(R)                            # [132, 8]

    T, q = data_true.size()                     # T=132 q=19
    _, M = z.size()                             # M=8

    z_gen = generate_free_trajectory(z, T, M, A, W, h)  # [132, 8]

    # exist hrf matrix
    H = tc.tensor(H, dtype=tc.float64)
    # generate new hrf matrix
    #matlab_eng = matlab.engine.start_matlab()
    #H = tc.tensor(matlab_eng.hrf_matrix(T, M, float(TR)), dtype=tc.float64)

    data_gen = output_mapping(z_gen, H, M, T, B_est, M_est, R)

    data_gen = data_gen.unsqueeze(0).numpy()  # (1, 1322, 64)
    data_true = data_true.unsqueeze(0).numpy()  # (1, 132, 64)

    # 全局变量 SMOOTHING=1  original CUTOFF=1500
    # SMOOTHING_SIGMA
    # FREQUENCY_CUTOFF in Hertz (assuming 1 time step to be 1 s)
    print('SMOOTHING:{}, CUTOFF:{}'.format(SMOOTHING, CUTOFF))

    pse_per_dim = power_spectrum_error_per_dim(x_gen=data_gen, x_true=data_true, smoothing=SMOOTHING, cutoff=CUTOFF)

    # AVERAGE
    pse = np.mean(pse_per_dim)

    printf('# AVG PSE {}'.format(pse))
    return pse, pse_per_dim



if __name__ == '__main__':

    setup_seed(1)
    PRINT = True

    X_DIM = 19
    # calculate MSE
    n_steps = 20
    # calculate KLx
    INITCONDS = 1
    # calculate PSE
    SMOOTHING = 3       # for PSE Noise Removal
    CUTOFF = -1         # CUTOFF = -1 NO CUTOFF

    data_root_path = 'D:\Projects_local\PLRNN_new\data_local_new_trianOUT\dataset_resample_result4python_combine\\test\\'
    data_file_name = 'INDI_4JXD2_normBEF_re1.6_X19_Z8_lam1000_rep5'
    data_file_path = data_root_path + data_file_name + '.mat'

    print('DATA ROOT PATH: {}'.format(data_file_path))
    assert os.path.exists(data_file_path)
    file_name = data_file_path.split('\\')[-1]

    model_data = loadmat(data_file_path)

    # Parameter Extract
    sample_id = model_data['sample_id'][0]          # Sample ID
    norm_method = model_data['norm_method'][0]      # norm_method normAFT/normBEF
    resample_tr = model_data['resample_tr'][0][0]   #
    roi_num = model_data['roi_num'][0][0]
    latent_z_dim = model_data['latent_z_dim'][0][0]
    lamda = model_data['lamda'][0][0]
    rep_ver = model_data['rep_ver'][0]

    Ezi = model_data['Ezi']             # (8, 132)
    M = model_data['M'][0][0]           # 8
    X = model_data['X']                 # (19, 132)
    q = model_data['q'][0][0]           # 19
    T = model_data['T'][0][0]           # 132
    R = model_data['R']                 # (132, 8) R+cov
    cov = model_data['cov']             # (132, 2)
    regions = np.array(model_data['regions']).squeeze(-1)   # 19 ['PrG']

    mu0 = model_data['mu0']             # (8, 1)
    W = model_data['W']                 # (8, 8)
    A = model_data['A']                 # (8, 8)
    h = model_data['h']                 # (8, 1)
    B_est = model_data['B_est']         # (19, 8)
    M_est = model_data['M_est']         # (19, 8)
    G_est = model_data['G_est']         # (19, 19)
    H = model_data['H']                 # (1056, 1056)
    Z_fr = model_data['Z_fr']           # (8, 132)
    X_fr_pre = model_data['X_fr_pre']   # (19, 132)

    # 3 metrics
    # pse ==> Power Spectrum Correlation
    # mse ==> Mean Squared Prediction Error A.K.A PE
    # klx ==> KL divergence D_stsp
    # metrics = ['mse', 'pse', 'klx']
    matlab_eng = matlab.engine.start_matlab()

    # calculate MSE
    mse_results = mse_evaluator(X, Ezi, A, W, h, B_est, M_est, R, resample_tr, matlab_eng, n_steps)
    print('# MSE-AVG = {}'.format(np.mean(mse_results)))
    print('-----------------------')

    # calculate KLx
    klx_results = klx_evaluator(X, Ezi, A, W, h, B_est, M_est, R, resample_tr, H)
    print('-----------------------')

    # calculate PSE
    pse_avg, pse_per_dim = pse_evaluator(X, Ezi, A, W, h, B_est, M_est, R, H)
    print('-----------------------')

    print('MODEL EVA END')
    print('-----------------------')




