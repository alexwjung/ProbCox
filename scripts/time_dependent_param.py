# Modules
# =======================================================================================================================

import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

import warnings
warnings.filterwarnings("ignore")

import probcox as pcox

dtype = torch.FloatTensor

np.random.seed(34)
torch.manual_seed(234)

# Loading Data
# =======================================================================================================================

veterans = pd.read_csv('/Users/alexwjung/projects/ProbCox_extensions/data/vets.csv')

surv = np.concatenate((np.asarray(veterans.iloc[:, 8])[:, None], np.asarray(veterans.iloc[:, 9])[:, None], np.asarray(veterans.iloc[:, 10])[:, None]), axis=1)

X = np.concatenate((np.asarray(veterans.iloc[:, 1])[:, None], np.asarray(veterans.iloc[:, 6])[:, None], np.asarray(veterans.iloc[:, 3])[:, None]), axis=1)

# inmdicator - R - https://cran.r-project.org/web/packages/survival/vignettes/timedep.pdf
'''
idx1 = (np.asarray(veterans.iloc[:, -1]) == 1).astype(int)
idx2 = (np.asarray(veterans.iloc[:, -1]) == 2).astype(int)
idx3 = (np.asarray(veterans.iloc[:, -1]) == 3).astype(int)
X = np.concatenate((np.asarray(veterans.iloc[:, 1])[:, None], np.asarray(veterans.iloc[:, 6])[:, None], np.asarray(veterans.iloc[:, 3])[:, None], (np.asarray(veterans.iloc[:, 3]) *  idx1)[:, None], (np.asarray(veterans.iloc[:, 3]) *  idx2)[:, None], (np.asarray(veterans.iloc[:, 5]) *  idx3)[:, None]), axis=1)
'''


surv = torch.from_numpy(surv).type(torch.FloatTensor)
X = torch.from_numpy(X).type(torch.FloatTensor)

total_obs = surv.shape[0]
batch_size = total_obs
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# Inference
# =======================================================================================================================
# Custom linear predictor - Here: simple linear combination
def predictor(data):
    theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.001).expand([data[1].shape[1], 1])).type(dtype)
    pred = torch.mm(data[1], theta)
    return(pred)

run = True
eta = 1.0
while run:
    run = False
    pyro.clear_param_store()
    m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=None)
    m.initialize(eta=eta, num_particles=5, rank=None)
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


# Summary
# =======================================================================================================================
for ii in range(X.shape[1]):
    if np.sign(out['theta'][0].detach().numpy()[:, 0][ii]) == np.sign(out['theta'][2].detach().numpy()[:, 0][ii]):
        print(out['theta'][1].detach().numpy()[:, 0][ii], '***')
    else:
        print(out['theta'][1].detach().numpy()[:, 0][ii], '')



pd.DataFrame(np.concatenate((surv.detach().numpy(), X.detach().numpy()), axis=1)).to_csv('/Users/alexwjung/Desktop/vets.csv')
