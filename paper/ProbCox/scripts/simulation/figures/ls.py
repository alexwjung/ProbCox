'''

Large-scale Case Simulation - Figure:

- combine results from individual simulation runs and produce summary figure

'''


# Modules
# =======================================================================================================================
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/paper/ProbCox')
#os.chdir('/nfs/research/gerstung/awj/projects/ProbCox/paper/ProbCox')

sim_name = 'sim_ls'

# Make
# =======================================================================================================================
I = 50000
P = 10
theta = np.asarray(pd.read_csv('./out/simulation/' + sim_name + '/theta.txt', header=None))

#=======================================================================================================================

for batchsize in ['48']:

    # empty file to write results into
    res = np.zeros((P, 6))
    res[:, 0] = theta[:, 0]

    theta_est = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(batchsize) + '_theta.txt', header=None, sep=';')
    theta_est = theta_est.dropna(axis=0)
    theta_est = theta_est.groupby(0).first().reset_index()
    theta_est = theta_est.iloc[:, :-1]
    assert theta_est.shape[0] == 50

    theta_est_lower = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(batchsize) + '_theta_lower.txt', header=None, sep=';')
    theta_est_lower = theta_est_lower.dropna(axis=0)
    theta_est_lower = theta_est_lower.groupby(0).first().reset_index()
    theta_est_lower = theta_est_lower.iloc[:, :-1]
    assert theta_est_lower.shape[0] == 50

    theta_est_upper = pd.read_csv('./out/simulation/' + sim_name + '/probcox' + str(batchsize) + '_theta_upper.txt', header=None, sep=';')
    theta_est_upper = theta_est_upper.dropna(axis=0)
    theta_est_upper = theta_est_upper.groupby(0).first().reset_index()
    theta_est_upper = theta_est_upper.iloc[:, :-1]
    assert theta_est_upper.shape[0] == 50

    theta_bound = theta_est_lower.merge(theta_est_upper, how='inner', on=0)
    theta_bound = theta_bound.merge(theta_est, how='inner', on=0)
    theta_est = np.asarray(theta_bound.iloc[:, -P:]).astype(float)
    theta_bound = theta_bound.iloc[:, :-P]
    theta_bound = np.asarray(theta_bound.iloc[:, 1:]).astype(float)


# Plot Settings
# =======================================================================================================================

plt.rcParams['font.size'] = 7
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
cm = 1/2.54
marker=['.', '2', '*', 's', 'h']

# Plot
# =======================================================================================================================
fig, ax = plt.subplots(1, 1, figsize=(6*cm, 6*cm), dpi=600)
for ii in range(25):
    ax.plot(theta, theta_est[ii, :], ls='', marker='.', ms=2, c='.6')
ax.plot(theta, theta_est[0, :], ls='', marker='.', ms=2, c='.6', label=r'$\hat{\theta}$')
ax.plot(theta, np.mean(theta_est, axis=0), ls='', marker='x', ms=3, c='.2', label=r'$\bar{\hat{\theta}}$')
ax.plot(theta, np.mean(theta_bound[:, :10], axis=0)[:, None], ls='', marker='_', ms=4, c='.2', label=r'$\bar{\hat{\theta}}_{0.975}$')
ax.plot(theta, np.mean(theta_bound[:, -10:], axis=0)[:, None], ls='', marker='_', ms=4, c='.2', label=r'$\bar{\hat{\theta}}_{0.025}$')
ax.set(xlim=(-2, 2), ylim=(-2, 2))
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\hat{\theta}$')
ax.set_yticks([-2, 0, 2])
ax.set_ylim([-2, 2])
ax.set_xticks([-2, 0, 2])
ax.set_xlim([-2, 2])


ax.plot(ax.get_xlim(), ax.get_ylim(), ls=':', color='black', linewidth=0.75)
ax.legend(frameon=False, prop={'size': 5}, loc='lower right')
#plt.show()
plt.savefig('./out/simulation/figures/ls.eps', bbox_inches='tight', dpi=600, transparent=True)
plt.savefig('./out/simulation/figures/ls.png', bbox_inches='tight', dpi=600, transparent=True)
plt.savefig('./out/simulation/figures/ls.pdf', bbox_inches='tight', dpi=600, transparent=True)
plt.close()
