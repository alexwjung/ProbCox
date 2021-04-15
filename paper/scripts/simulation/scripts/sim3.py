import os
import sys
import subprocess
import tqdm
import importlib
os.chdir('/nfs/nobackup/gerstung/awj/projects/ProbCox/')

import pandas as pd
import numpy as np

from multiprocessing import Pool

import torch
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader

import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 22
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')

import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

import simulation as sim
importlib.reload(sim)

# Setup
# -----------------------------------------------------------------------------------------------------------------------------

sim_name = 'sim3'

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(87)
torch.manual_seed(34)

# Dataloader Settings:
# -----------------------------------------------------------------------------------------------------------------------------

class sim_dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self):
        """
        """

    def __len__(self):
        return(141)

    def __getitem__(self, ii):
        d = torch.load('./tmp/' + str(ii))
        surv = d['surv']
        X = d['X']
        return(surv, X)
    
TVC_data = sim_dataset()
dataloader = DataLoader(TVC_data, batch_size=1, num_workers=4, prefetch_factor=1, persistent_workers=True)

n = 0 
for _, d in tqdm.tqdm(iter(enumerate(dataloader))):
    n += d[0].shape[1]



# Inference:
# -----------------------------------------------------------------------------------------------------------------------------
'''
e=0
for _, __input__ in tqdm.tqdm(enumerate(dataloader)):
    e += torch.sum(__input__[0][0, :, -1])
e
'''

total_obs = 140 * 2048
total_events = 5230
batchsize = 2048
sampling_proportion = [total_obs, batchsize, total_events, None]

def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(3, loc=0, scale=0.15).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)
run = True
eta=0.9
while run:
    run = False
    pyro.clear_param_store()
    m = pcox.PCox(predictor=predictor, sampling_proportion=sampling_proportion)
    m.initialize(eta=eta, rank=30, num_particles=25)
    loss=[0]
    for __ in range(25): 
        for _, __input__ in tqdm.tqdm(enumerate(dataloader)):
            loss.append(m.infer(data=(torch.squeeze(__input__[0]), torch.squeeze(__input__[1]))))
            if loss[-1] != loss[-1]:
                break   
        plt.semilogy(loss)
        plt.show()
        plt.close()
    

# plot
g = m.return_guide()
out = g.quantiles([0.025, 0.5, 0.975])

theta = pd.read_csv('./output/simulation/sim3/theta.txt', header=None)
theta = np.asarray(theta).astype(float)
fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=900)
plt.errorbar(theta[:, 0], out['theta'][1].detach().numpy()[:, 0], yerr=(out['theta'][1].detach().numpy()[:, 0] - out['theta'][0].detach().numpy()[:, 0], out['theta'][2].detach().numpy()[:, 0]- out['theta'][1].detach().numpy()[:, 0]),  ls='', c=".3", capsize=2, capthick=0.95, elinewidth=0.95)
ax.plot(theta[:, 0], out['theta'][1].detach().numpy()[:, 0], ls='', c=".3", marker='x', ms=2)
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3", linewidth=0.75)
ax.set_yticks([-1.5, 0, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_xticks([-1.5, 0, 1.5])
ax.set_xlim([-1.5, 1.5])
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\hat{\theta}$')
plt.savefig('./output/simulation/figures/sim3.eps', bbox_inches='tight')


# save model fit 
with open('./output/simulation/' + sim_name + '/probcox_theta.txt', 'w') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
with open('./output/simulation/' + sim_name + '/probcox_theta_lower.txt', 'w') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
with open('./output/simulation/' + sim_name + '/probcox_theta_upper.txt', 'w') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
with open('./output/simulation/' + sim_name + '/probcox_theta_se.txt', 'w') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in pyro.get_param_store()['AutoLowRankMultivariateNormal.scale'].detach().numpy().tolist()]))
    write_out.write('\n')  

