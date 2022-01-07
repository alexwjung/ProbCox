'''

Standard Case Simulation - Case 1 - Table:

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

sim_name = 'sim_sc1'

# Make
# =======================================================================================================================
I = 1000
P = 6
theta = np.asarray(pd.read_csv('./out/simulation/' + sim_name + '/theta.txt', header=None))

# Overall Parameters
# =======================================================================================================================

N_obs = pd.read_csv('./out/simulation/' + sim_name + '/N_obs.txt', sep=';', header=None)

print(np.min(N_obs.iloc[:, 1]), np.median(N_obs.iloc[:, 1]), np.max(N_obs.iloc[:, 1]))
print(np.min(1-N_obs.iloc[:, 2]/I), np.median(1-N_obs.iloc[:, 2]/I), np.max(1-N_obs.iloc[:, 2]/I))
print(np.min(N_obs.iloc[:, 3]), np.median(N_obs.iloc[:, 3]), np.max(N_obs.iloc[:, 3]))

# ProbCox
# =======================================================================================================================

for batchsize in ['64', '128', '256', '512', 'full']:

    # empty file to write results into
    res = np.zeros((P, 6))
    res[:, 0] = theta[:, 0]

    theta_est = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(batchsize) + '_theta.txt', header=None, sep=';')
    theta_est = theta_est.dropna(axis=0)
    theta_est = theta_est.groupby(0).first().reset_index()
    theta_est = theta_est.iloc[:, :-1]
    assert theta_est.shape[0] == 200

    theta_est_lower = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(batchsize) + '_theta_lower.txt', header=None, sep=';')
    theta_est_lower = theta_est_lower.dropna(axis=0)
    theta_est_lower = theta_est_lower.groupby(0).first().reset_index()
    theta_est_lower = theta_est_lower.iloc[:, :-1]
    assert theta_est_lower.shape[0] == 200

    theta_est_upper = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(batchsize) + '_theta_upper.txt', header=None, sep=';')
    theta_est_upper = theta_est_upper.dropna(axis=0)
    theta_est_upper = theta_est_upper.groupby(0).first().reset_index()
    theta_est_upper = theta_est_upper.iloc[:, :-1]
    assert theta_est_upper.shape[0] == 200

    theta_bound = theta_est_lower.merge(theta_est_upper, how='inner', on=0)
    theta_bound = theta_bound.merge(theta_est, how='inner', on=0)
    theta_est = np.asarray(theta_bound.iloc[:, -P:]).astype(float)
    theta_bound = theta_bound.iloc[:, :-P]
    theta_bound = np.asarray(theta_bound.iloc[:, 1:]).astype(float)

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
    pd.DataFrame(res).to_csv('./out/simulation/tables/' + sim_name + '_ProbCox_' + batchsize + '.csv')


# R Cox
# =======================================================================================================================

res = np.zeros((P, 6))
res[:, 0] = theta[:, 0]
theta_est = pd.read_csv('./out/simulation/' + sim_name + '/R_theta.txt', header=None, sep=';')
theta_est = theta_est.dropna(axis=0)
theta_est = theta_est.groupby(0).first().reset_index()
assert theta_est.shape[0] == 200

theta_se = pd.read_csv('./out/simulation/' + sim_name + '/R_se.txt', header=None, sep=';')
theta_se = theta_se.dropna(axis=0)
theta_se = theta_se.groupby(0).first().reset_index()
assert theta_se.shape[0] == 200


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
pd.DataFrame(res).to_csv('./out/simulation/tables/' + sim_name + '_R' + '.csv')


print('finished')
