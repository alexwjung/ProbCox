'''

High Dimensional Case2 Simulation:

- combine results from individual simulation runs and produce summary table

- headings:
$\theta$   $\bar{\hat{\theta}}$ 	$\overline{\sigma_{\hat{\theta}}}$	$RMSE$ 	$\overline{HPD}_{95\%}$	$Coverage_{95\%}$

'''


# Modules
# =======================================================================================================================
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')
os.chdir('/nfs/research/gerstung/awj/projects/ProbCox/paper/ProbCox')

sim_name = 'sim_hd2'

# Function
# =======================================================================================================================
def custom_mean(X, W, col_idx):
    '''
    - average for paramters of an array selcted by an indexing matrix

    X :: array to apply mean along axis=0
    W :: indexing which elements to use for mean computatiuon
    col_idx :: indexing the columns where W is applied - otherwise standard mean without selecting elements
    '''
    m = []
    assert X.shape == W.shape
    N, M = X.shape

    for jj in range(M):
        if col_idx[jj] == True:
            m.append(np.mean(X[W[:, jj], jj]))
        else:
            m.append(np.mean(X[:, jj]))
    return(np.asarray(m))


def custom_var(X, W, col_idx):
    '''s
    - variance for paramters of an array selcted by an indexing matrix

    X :: array to apply variance along axis=0
    W :: indexing which elements to use for variance computatiuon
    col_idx :: indexing the columns where W is applied - otherwise standard mean without selecting elements
    '''
    m = []
    assert X.shape == W.shape
    N, M = X.shape

    for jj in range(M):
        if col_idx[jj] == True:
            m.append(np.var(X[W[:, jj], jj]))
        else:
            m.append(np.var(X[:, jj]))
    return(np.asarray(m))

# Make
# =======================================================================================================================
I = 750
P = 1000
theta = np.concatenate((np.asarray([-0.5, 0.7, 1.2, 0.65, -0.9, 1.4, 0.2, -0.4, -1.3, 0.1]), np.zeros((990,))))[:, None]

# Overall Parameters
# ======================================================================================================================
N_obs = pd.read_csv('./out/simulation/' + sim_name + '/N_obs.txt', sep=';', header=None)

print(np.min(N_obs.iloc[:, 1]), np.median(N_obs.iloc[:, 1]), np.max(N_obs.iloc[:, 1]))
print(np.min(1-N_obs.iloc[:, 2]/I), np.median(1-N_obs.iloc[:, 2]/I), np.max(1-N_obs.iloc[:, 2]/I))
print(np.min(N_obs.iloc[:, 3]), np.median(N_obs.iloc[:, 3]), np.max(N_obs.iloc[:, 3]))

# ProbCox
# =======================================================================================================================

for suffix in ['rank5', 'rank50']:

    # empty file to write results into
    res = np.zeros((P, 7))
    res[:, 0] = theta[:, 0]

    theta_est = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(suffix) + '_theta.txt', header=None, sep=';')
    theta_est = theta_est.dropna(axis=0)
    theta_est = theta_est.groupby(0).first().reset_index()
    theta_est = theta_est.iloc[:, :-1]
    assert theta_est.shape[0] == 200

    theta_est_lower = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(suffix) + '_theta_lower.txt', header=None, sep=';')
    theta_est_lower = theta_est_lower.dropna(axis=0)
    theta_est_lower = theta_est_lower.groupby(0).first().reset_index()
    theta_est_lower = theta_est_lower.iloc[:, :-1]
    assert theta_est_lower.shape[0] == 200

    theta_est_upper = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(suffix) + '_theta_upper.txt', header=None, sep=';')
    theta_est_upper = theta_est_upper.dropna(axis=0)
    theta_est_upper = theta_est_upper.groupby(0).first().reset_index()
    theta_est_upper = theta_est_upper.iloc[:, :-1]
    assert theta_est_upper.shape[0] == 200

    theta_bound = theta_est_lower.merge(theta_est_upper, how='inner', on=0)
    theta_bound = theta_bound.merge(theta_est, how='inner', on=0)
    theta_est = np.asarray(theta_bound.iloc[:, -P:]).astype(float)
    theta_bound = theta_bound.iloc[:, :-P]
    theta_bound = np.asarray(theta_bound.iloc[:, 1:]).astype(float)

    theta_est_lower = np.asarray(theta_est_lower.iloc[:, 1:])
    theta_est_upper = np.asarray(theta_est_upper.iloc[:, 1:])

    W = np.sign(theta_est_lower) == np.sign(theta_est_upper) # non zero parameters estimates (based on HPD95%)
    col_idx = np.logical_and(np.squeeze(theta != 0), np.sum(W, axis=0) > 5) # true non-zero parameters


    res[:, 1] = custom_mean(theta_est, W, col_idx)
    res[:, 2] = np.sqrt(custom_var(theta_est, W, col_idx))
    res[:, 3] = np.sqrt(custom_mean((theta_est - theta[:, 0][None, :])**2, W, col_idx))

    res[:, 4] = custom_mean(theta_bound[:, -P:] - theta_bound[:, :P], W, col_idx)

    res[:, 5] = custom_mean(np.logical_and(np.squeeze(theta)[None, :] >= theta_bound[:, :P], np.squeeze(theta)[None, :] <= theta_bound[:, -P:])
, W, col_idx)

    res[:, 6] = np.mean(W, axis=0)

    res = np.round(res, 2)

    pd.DataFrame(res).to_csv('./out/simulation/tables/' + sim_name + '_ProbCox_' + suffix + '.csv')
    pd.DataFrame(res[:10, :]).to_csv('./out/simulation/tables/' + sim_name + '_ProbCox_main_' + suffix + '.csv')

    theta_est_lower = theta_bound[:, :1000]
    theta_est_upper = theta_bound[:, 1000:]

    pd.DataFrame(np.concatenate((np.round(np.mean(np.sum(np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :]), axis=1)))[None, None], np.round(np.mean(np.sum((np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :])) * np.squeeze(theta != 0)[None, :], axis=1)))[None, None]), axis=1)).to_csv('./out/simulation/tables/' + sim_name + '_ProbCox_main_add_' + suffix + '.csv')


# R Cox
# =======================================================================================================================
for suffix in ['R_lasso_theta', 'R_lasso_theta_1se', 'R_Alasso1_theta', 'R_Alasso1_theta_1se', 'R_Alasso2_theta', 'R_Alasso2_theta_1se', 'R_SCAD_theta', 'R_MCP_theta']:

    res = np.zeros((P, 7))
    res[:, 0] = theta[:, 0]
    theta_est = pd.read_csv('./out/simulation/' + sim_name + '/' + suffix + '.txt', header=None, sep=';')
    theta_est = theta_est.dropna(axis=0)
    theta_est = theta_est.groupby(0).first().reset_index()
    theta_est = np.asarray(theta_est.iloc[:, 1:])
    assert theta_est.shape[0] == 200


    W = theta_est!=0 # non zero parameters estimates (based on HPD95%)
    col_idx = np.logical_and(np.squeeze(theta != 0), np.sum(W, axis=0) > 5) # true non-zero parameters

    res[:, 1] = custom_mean(theta_est, W, col_idx)
    res[:, 2] = np.sqrt(custom_var(theta_est, W, col_idx))
    res[:, 3] = np.sqrt(custom_mean((theta_est - theta[:, 0][None, :])**2, W, col_idx))

    res[:, 6] = np.mean(W, axis=0)

    res = np.round(res, 2)

    pd.DataFrame(res).to_csv('./out/simulation/tables/' + sim_name + '_' + suffix + '_all' + '.csv')
    res = pd.DataFrame(res[:10, :])
    res.iloc[:, 4] = '-'
    res.iloc[:, 5] = '-'
    res.to_csv('./out/simulation/tables/' + sim_name + '_' +  suffix + '_main' + '.csv')

    pd.DataFrame(np.concatenate((np.round(np.mean(np.sum(theta_est != 0, axis=1)))[None, None], np.round(np.mean(np.sum((theta_est != 0) * np.squeeze(theta != 0)[None, :], axis=1)))[None, None]), axis=1)).to_csv('./out/simulation/tables/' + sim_name + '_' + suffix + '_add' + '.csv')

print('finished')
