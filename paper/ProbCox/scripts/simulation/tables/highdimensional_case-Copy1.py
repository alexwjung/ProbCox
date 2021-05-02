'''

High Dimensional Case Simulation:

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

os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

sim_name = 'sim_hd'

# Function
# =======================================================================================================================
def custom_mean(X, W, col_idx):
    '''
    - mean to compute the average for paramters that are identified as non-zero.
    
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
        
# Make
# =======================================================================================================================
I = 1000
P = 10000
theta = np.asarray(pd.read_csv('./out/' + sim_name + '/theta.txt', header=None))



# Overall Parameters
# =======================================================================================================================

N_obs = pd.read_csv('./out/' + sim_name + '/N_obs.txt', sep=';', header=None)
print('Mean number of intervall observations: ', np.mean(N_obs.iloc[:, 1]))
print('Mean number of censorship: ', 1 - (np.mean(N_obs.iloc[:, 2])/I))

# ProbCox
# =======================================================================================================================

for suffix in ['rank5', 'rank50', 'rank50_b1024']:
    
    # empty file to write results into
    res = np.zeros((P, 7))
    res[:, 0] = theta[:, 0]

    theta_est = pd.read_csv('./out/' + sim_name + '/probcox' + str(suffix) + '_theta.txt', header=None, sep=';')
    theta_est = theta_est.dropna(axis=0)
    theta_est = theta_est.groupby(0).first().reset_index()
    theta_est = theta_est.iloc[:, :-1]
    assert theta_est.shape[0] == 100

    theta_est_lower = pd.read_csv('./out/' + sim_name + '/probcox' + str(suffix) + '_theta_lower.txt', header=None, sep=';')
    theta_est_lower = theta_est_lower.dropna(axis=0)
    theta_est_lower = theta_est_lower.groupby(0).first().reset_index()
    theta_est_lower = theta_est_lower.iloc[:, :-1]
    assert theta_est_lower.shape[0] == 100

    theta_est_upper = pd.read_csv('./out/' + sim_name + '/probcox' + str(suffix) + '_theta_upper.txt', header=None, sep=';')
    theta_est_upper = theta_est_upper.dropna(axis=0)
    theta_est_upper = theta_est_upper.groupby(0).first().reset_index()
    theta_est_upper = theta_est_upper.iloc[:, :-1]
    assert theta_est_upper.shape[0] == 100

    theta_bound = theta_est_lower.merge(theta_est_upper, how='inner', on=0)
    theta_bound = theta_bound.merge(theta_est, how='inner', on=0)
    theta_est = np.asarray(theta_bound.iloc[:, -P:]).astype(float)
    theta_bound = theta_bound.iloc[:, :-P]
    theta_bound = np.asarray(theta_bound.iloc[:, 1:]).astype(float)

    
    W = np.sign(theta_est_lower) == np.sign(theta_est_upper) # non zero parameters estimates (based on HPD95%)
    col_idx = np.logical_and(np.squeeze(theta != 0), np.sum(W, axis=0) > 5) # true non-zero parameters


    res[:, 1] = np.mean(theta_est, axis=0)
    res[:, 2] = np.sqrt(np.var(theta_est, axis=0))
    res[:, 3] = np.sqrt(np.mean((theta_est - theta[:, 0][None, :])**2, axis=0))

    ll = []
    for ii in range(P):
        ll.append(np.mean(theta_bound[:, ii+P] - theta_bound[:, ii]))
    res[:, 4] = np.asarray(ll)    

    ll = []
    for ii in range(P):
        ll.append(np.sum(np.logical_and(theta[ii]>=theta_bound[:, ii], theta[ii]<=theta_bound[:, ii+P]))/theta_bound.shape[0])
    res[:, 5] = np.asarray(ll)       
    
    res = np.round(res, 2)
    
    
    pd.DataFrame(np.concatenate((res[:10, :], res[5000:5010, :])))
    
    
    #pd.DataFrame(res).to_csv('./out/tables/' + sim_name + '_ProbCox_' + suffix + '.csv')
    #pd.DataFrame(np.concatenate((res[:10, :], res[5000:5010, :]))).to_csv('./out/tables/' + sim_name + '_ProbCox_main_' + suffix + '.csv')
    
    theta_est_lower = theta_bound[:, :10000]
    theta_est_upper = theta_bound[:, 10000:]    
    print(suffix+ ' - P: ', np.mean(np.sum(np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :]), axis=1)))
    print(suffix+ ' - seP: ', np.sqrt(np.var(np.sum(np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :]), axis=1))))
    print(suffix+ ' - FP: ', np.mean(np.sum((np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :])) * np.squeeze(theta == 0)[None, :], axis=1)))


    
# R Cox
# =======================================================================================================================

res = np.zeros((P, 6))
res[:, 0] = theta[:, 0]
theta_est = pd.read_csv('./out/' + sim_name + '/R_theta.txt', header=None, sep=';')
theta_est = theta_est.dropna(axis=0)
theta_est = theta_est.groupby(0).first().reset_index()
assert theta_est.shape[0] == 100

theta_se = pd.read_csv('./out/' + sim_name + '/R_se.txt', header=None, sep=';')
theta_se = theta_se.dropna(axis=0)
theta_se = theta_se.groupby(0).first().reset_index()
assert theta_se.shape[0] == 100


theta_mat = theta_est.merge(theta_se, how='inner', on=0)
theta_mat = np.asarray(theta_mat.iloc[:, 1:]).astype(float)
print(theta_mat.shape[0])

theta_est = theta_mat[:, :P]
theta_se = theta_mat[:, P:]

res[:, 1] = np.mean(theta_est, axis=0)
res[:, 2] = np.sqrt(np.var(theta_est, axis=0))
res[:, 3] = np.sqrt(np.mean((theta_est - theta[:, 0][None, :])**2, axis=0))

theta_est_lower = theta_est - 1.96*theta_se
theta_est_upper = theta_est + 1.96*theta_se

theta_bound = np.concatenate((theta_est_lower, theta_est_upper), axis=1)

ll = []
for ii in range(P):
    ll.append(np.mean(theta_bound[:, ii+P] - theta_bound[:, ii]))
res[:, 4] = np.asarray(ll)    

ll = []
for ii in range(P):
    ll.append(np.sum(np.logical_and(theta[ii]>=theta_bound[:, ii], theta[ii]<=theta_bound[:, ii+P]))/theta_bound.shape[0])
res[:, 5] = np.asarray(ll)       

res = np.round(res, 2)
pd.DataFrame(res).to_csv('./out/tables/' + sim_name + '_R' + '.csv')
pd.DataFrame(np.concatenate((res[:10, :], res[5000:5010, :]))).to_csv('./out/tables/' + sim_name + '_R_main' + '.csv')

theta_est_lower = theta_bound[:, :10000]
theta_est_upper = theta_bound[:, 10000:]    
print('R - P: ', np.mean(np.sum(np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :]), axis=1)))
print('R - seP: ', np.sqrt(np.var(np.sum(np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :]), axis=1))))
print('R - FP: ', np.mean(np.sum((np.sign(theta_est_lower[:, :]) == np.sign(theta_est_upper[:, :])) * np.squeeze(theta == 0)[None, :], axis=1)))


print('finished')