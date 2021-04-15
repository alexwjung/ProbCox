'''
Plot schematic of the simulation algorythm
'''

# Modules
# -----------------------------------------------------------------------------------------------------------------------------
import os
import sys
import glob
import subprocess
import tqdm
import importlib
os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Functions
# -----------------------------------------------------------------------------------------------------------------------------

def forward_fill(x):
    for ii in range(1, x.shape[0]):
        if np.sum(x[ii, :]) == 0:
            x[ii, :] = x[ii-1, :]
    return(x)

def logisitc(x):
    return(1/(1+np.exp(-x)))

def logit(x):
    return(np.log(x/(1-x)))

# Figure settings
# -----------------------------------------------------------------------------------------------------------------------------
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Figure
# -----------------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(8.27/1, 11.69/2), dpi=300, sharex=True)  
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.01)


tlambda0 = np.asarray([0, 200, 400, 500, 800, 850, 950, 1100, 1300, 1500, 1700])
lambda0 = np.asarray([5, 10, 20, 25, 50, 60, 90, 140, 120, 90, 50])/10000

for ii in range(10):  
    ax[0].plot([tlambda0[ii], tlambda0[ii+1]], [lambda0[ii], lambda0[ii]], c='.5')
    if ii < 9:
        ax[0].scatter(tlambda0[ii+1], lambda0[ii], s=25, facecolors='none', edgecolors='.5')
    ax[0].scatter(tlambda0[ii], lambda0[ii], s=25, c='.5')
    ax[0].axvline(tlambda0[ii], ymin=0, ymax=lambda0[ii]/0.015, c='.4', ls=':', linewidth=1)
ax[0].set_ylim(0, 0.015)
ax[0].set_ylabel(r'$\alpha_0$')
ax[0].set_yticklabels([])


tx = np.asarray([0, 300, 700, 1000, 1400, 1700])
x = np.asarray([0.1, 0.4, -0.2, 0.1, 0.3, 0.2])
for ii in range(5):
    ax[1].plot([tx[ii], tx[ii+1]], [x[ii], x[ii]], c='.5')
    if ii < 4:
        ax[1].scatter(tx[ii+1], x[ii], s=25, facecolors='none', edgecolors='.5')
    ax[1].scatter(tx[ii], x[ii], s=25, c='.5')
    ax[1].axvline(tx[ii], ymin=0, ymax=(x[ii]+0.25)/0.7, c='.3', ls=(0, (5, 10)), linewidth=1)
for ii in range(10):    
    ax[1].axvline(tlambda0[ii], c='.4', ls=':', linewidth=1)
    
ax[1].axhline(0, c='r', ls='--', linewidth=0.25)
ax[1].text(1800, -0.02, r'$0$')
ax[1].set_ylim(-0.25, 0.45)
ax[1].set_yticklabels([])
ax[1].set_ylabel(r'$Z(t)\mathbf{\theta}$')

basehaz = np.copy(lambda0)[:, None]
t_lambda0 = np.copy(tlambda0)
X = np.copy(x)[:, None]
t_cov= np.copy(tx)

#expand
basehaz = np.concatenate((basehaz, np.zeros((t_cov.shape[0], 1))))
X = np.concatenate((np.zeros((t_lambda0.shape[0], 1)), X))
t = np.concatenate((t_lambda0, t_cov))
idx_sort = np.argsort(t)
t = t[idx_sort]
basehaz = forward_fill(basehaz[idx_sort])
X = forward_fill(X[idx_sort])
        
# collapse
X = np.concatenate([np.sum(X[t==ii, :], axis=0)[None, :] for ii in np.unique(t)])
basehaz = np.concatenate([np.sum(basehaz[t==ii, :], axis=0)[None, :] for ii in np.unique(t)])
t = np.unique(t)
t
# artifical event
t = t[:-2]
t[-1] = 1450

# survial data
H = logisitc(X + logit(basehaz)) 

for ii in range(12):
    ax[2].plot([t[ii], t[ii+1]], [H[ii], H[ii]], c='.5')
    if ii != 11:
        ax[2].scatter(t[ii+1], H[ii], s=25, facecolors='none', edgecolors='.5')
    ax[2].scatter(t[ii], H[ii], s=25, c='.5')
for ii in range(5):    
    ax[2].axvline(tx[ii], c='.3', ls=(0, (5, 10)), linewidth=1)
for ii in range(9):    
    ax[2].axvline(tlambda0[ii], c='.4', ls=':', linewidth=1)

ax[2].axvline(t[12], ymin=0, ymax=H[11]/0.016, c='#0b64e0', ls='-', linewidth=1)
ax[2].text(1475, 0.0074, r'$p(t) > U(0, 1)$')
ax[2].text(1475, 0.0056, r'$\rightarrow event$')
ax[2].set_ylim(0.0, 0.016)
ax[2].set_ylabel(r'$p(t)$') 
ax[2].set_xlabel(r'$Time$')
ax[2].set_xticklabels([])
ax[2].set_yticklabels([])
ax[2].set_xticks([])

plt.savefig('./output/simulation/figures/schematic.eps', bbox_inches='tight')


