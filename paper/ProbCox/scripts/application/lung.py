'''
Analysing the lung data from the R - survival package
'''

# Modules
# =======================================================================================================================
import os
import sys
import shutil
import subprocess
import tqdm

import numpy as np
import pandas as pd

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.infer import SVI, Trace_ELBO

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import probcox as pcox

dtype = torch.FloatTensor

np.random.seed(7832)
torch.manual_seed(64)

os.chdir('/Users/alexwjung/Documents/ProbCox/paper/ProbCox') # path to paper folder ProbCox

# Loading Data
# =======================================================================================================================

lung = pd.read_csv('./data/application/lung.csv', sep=',')
surv = np.asarray(lung[['time', 'status']])
surv = np.concatenate((np.zeros((surv.shape[0], 1)), surv), axis=1)
X = np.asarray(lung.iloc[:, 3:])

surv = torch.from_numpy(surv).type(torch.FloatTensor)
X = torch.from_numpy(X).type(torch.FloatTensor)

total_obs = surv.shape[0]
batch_size = 72
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# Inference
# =======================================================================================================================

run = True
eta = 1.0
while run:
    run = False
    pyro.clear_param_store()
    m = pcox.PCox(sampling_proportion=sampling_proportion)
    m.initialize(eta=eta, num_particles=5)
    loss=[0]
    for ii in tqdm.tqdm(range((10000))):
        idx = np.random.choice(range(surv.shape[0]), batch_size, replace=False)
        data=[surv[idx], X[idx]]
        if torch.sum(data[0][:, -1]) > 0:
            loss.append(m.infer(data=data))
        if loss[-1] != loss[-1]:
            eta = eta * 0.5
            run=True
            break
    g = m.return_guide()
    out = g.quantiles([0.025, 0.5, 0.975])
plt.semilogy(loss)

# Prepare summary tables
# =======================================================================================================================

a = np.round(out['theta'][1].detach().numpy()[:, 0], 2)
b = np.round(torch.diag(pyro.get_param_store()['AutoMultivariateNormal.scale_tril']).detach().numpy(), 2)
c =np.sign(out['theta'][0].detach().numpy()) == np.sign(out['theta'][2].detach().numpy())
for ii in range(X.shape[1]):
    if c[ii]:
        sig = '*'
    else:
        sig = ''
    print(str(a[ii]) + sig + ', (' + str(b[ii]) + ')')

ci = pcox.metrics(surv=surv.numpy(), linpred=torch.mm(X, out['theta'][1].detach()).numpy(), processes=4).concordance()[None]
print(ci)

np.savetxt('./out/application/lung/concordance.txt', ci)
np.savetxt('./out/application/lung/se.txt', torch.diag(pyro.get_param_store()['AutoMultivariateNormal.scale_tril']).detach().numpy())

with open('./out/application/lung/theta_lower.txt', 'a') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
with open('./out/application/lung/theta.txt', 'a') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
with open('./out/application/lung/theta_upper.txt', 'w') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
