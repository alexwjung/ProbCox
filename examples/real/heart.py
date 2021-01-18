import os
import sys
import tqdm
import importlib
import config
os.chdir(config.ROOT_DIR)

import pandas as pd
import numpy as np

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.mcmc import MCMC, NUTS

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')
import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

# -----------------------------------------------------------------------------------------------------------------------------
# importing csv data
heart = pd.read_csv('./data/real/heart.csv', sep=';')
surv = np.asarray(heart[['start', 'stop', 'event']])
X = np.asarray(heart[['age', 'year', 'surgery', 'transplant']])
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# transforming them to tensors
surv = torch.from_numpy(surv).type(torch.FloatTensor)
X = torch.from_numpy(X).type(torch.FloatTensor)

total_obs = surv.shape[0]
batch_size = 40
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# -----------------------------------------------------------------------------------------------------------------------------
pyro.clear_param_store()
m = pcox.PCox(sampling_proportion=sampling_proportion)
m.initialize(eta=1.0)
loss=[]
#idx = np.random.choice(range(surv.shape[0]), 30, replace=False)
for ii in tqdm.tqdm(range((10000))):
    idx = np.random.choice(range(surv.shape[0]), batch_size, replace=False)
    data=[surv[idx], X[idx]]
    loss.append(m.infer(data=data))
plt.semilogy(loss)
theta_est = pyro.get_param_store()['AutoMultivariateNormal.loc'].detach()
print(pcox.concordance(surv.detach(), torch.mm(X, (theta_est[:, None]).detach())))
for key, value in pyro.get_param_store().items():
    print(f"{key}:\n{value}\n")


# -----------------------------------------------------------------------------------------------------------------------------
# %% markdown
# # R - Code
# - code to fit the same model in R
# - to compare model estimates
# %% markdown
# rm(list=ls())
# library(survival)
# library(stargazer)
#
# data("heart")
# heart
#
# heart <- heart[is.finite(rowSums(colon)),]
#
# names(heart)
# m1 = coxph(Surv(start, stop, event) ~age  + year + surgery + transplant, data=heart)
# summary(m1)
