
# -----------------------------------------------------------------------------------------------------------------------------
import os
import sys
import glob
import subprocess
import tqdm
import importlib
os.chdir('/nfs/research1/gerstung/sds/sds-ukb-cancer/projects/ProbCox/')

import pandas as pd
import numpy as np

from multiprocessing import Pool

import torch
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader, Sampler

import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------------


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('/nfs/nobackup/gerstung/awj/projects/ProbCox/ProbCox/')

import probcox as pcox 
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)

# Setup
# -----------------------------------------------------------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor
#dtype=torch.cuda.FloatTensor

np.random.seed(87)
torch.manual_seed(34)


ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'
# -----------------------------------------------------------------------------------------------------------------------------

landmarks = torch.load(ROOT_DIR + 'projects/ProbCox/output/landmark')

    
fig, ax = plt.subplots(1, 2, figsize=(8.27, 11.69/4), dpi=300)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.40, hspace=None)


# ROC plot
fpr={}
tpr={}
roc_auc={}
fpr["train"], tpr["train"], _ = roc_curve(landmarks['train'][:, 1], landmarks['train'][:, 2])
roc_auc["train"] = auc(fpr["train"], tpr["train"])

fpr["valid"], tpr["valid"], _ = roc_curve(landmarks['valid'][:, 1], landmarks['valid'][:, 2])
roc_auc["valid"] = auc(fpr["valid"], tpr["valid"])

fpr["test"], tpr["test"], _ = roc_curve(landmarks['test'][:, 1], landmarks['test'][:, 2])
roc_auc["test"] = auc(fpr["test"], tpr["test"])

ax[0].plot(fpr['train'], tpr['train'], color='.9',
         lw=2, label='ROC train (%0.2f)' % roc_auc['train'], linestyle=':')
ax[0].plot(fpr['valid'], tpr['valid'], color='.6',
         lw=2, label='ROC valid (%0.2f)' % roc_auc['valid'], linestyle='-.')
ax[0].plot(fpr['test'], tpr['test'], color='#0b64e0',
         lw=2, label='ROC test (%0.2f)' % roc_auc['test'], linestyle='-')
ax[0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
ax[0].set_xlim([0.0, 1.0])
ax[0].set_ylim([0.0, 1.05])
ax[0].set_xlabel(r'$False \hspace{0.1cm}Positive\hspace{0.1cm} Rate$')
ax[0].set_ylabel(r'$True \hspace{0.1cm}Positive\hspace{0.1cm} Rate$')
ax[0].legend(loc="lower right", frameon=False, prop={'size': 9})


# KM plot
pp = np.percentile(landmarks['train'][:, 2], [25, 75, 90, 99])
idx = landmarks['test'][:, 2]>=pp[-1]
tt, km = pcox.KM(times=landmarks['test'][idx, 0], events=landmarks['test'][idx, 1])   
ax[1].step(tt, km, c='.1', label=r'$>= q_{0.99}$')

idx = landmarks['test'][:, 2]>=pp[-2]
tt, km = pcox.KM(times=landmarks['test'][idx, 0], events=landmarks['test'][idx, 1])   
ax[1].step(tt, km, c='.4', label=r'$>= q_{0.90}$')

idx = landmarks['test'][:, 2]>=pp[-3]
tt, km = pcox.KM(times=landmarks['test'][idx, 0], events=landmarks['test'][idx, 1])   
ax[1].step(tt, km, c='.7', label=r'$>= q_{0.75}$')

idx = landmarks['test'][:, 2]<=pp[-4]
tt, km = pcox.KM(times=landmarks['test'][idx, 0], events=landmarks['test'][idx, 1])   
ax[1].step(tt, km, c='.9', label=r'$<= q_{0.25}$')

tt, km = pcox.KM(times=landmarks['test'][:, 0], events=landmarks['test'][:, 1])   
ax[1].step(tt, km, c='red', label=r'all', ls=':')

ax[1].set_xlim(0, 3650)
ax[1].set_xticks([0, 1825, 3650])

ax[1].set_xlabel(r'$Time$')
ax[1].set_ylabel(r'$Survival \hspace{0.1cm} Probability$')
ax[1].legend(loc="lower left", frameon=False, prop={'size': 9})


#plt.show()
#plt.close()
plt.savefig(ROOT_DIR + 'projects/ProbCox/output/pred_plot.eps', bbox_inches='tight')
