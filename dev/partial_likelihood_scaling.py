# import the necessary modules
import sys
import os
import torch
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# set root directory
# path to the folder
dir_root = '/Users/alexwjung/Desktop/TensorCox_/'
os.chdir(dir_root)

# appends the path to the COX script 
sys.path.append(dir_root + 'TensorCox/')

# import COX model
from TensorCox import loglikelihood
from TensorCox import Fisher
from metrics import concordance
from metrics import RMSE
from dataloader import CSV_Dataset
from dataloader import ToTensor
from dataloader import custom_collate

torch.manual_seed(7)
np.random.seed(7)


# Data 
colon = pd.read_csv('data/colon.csv', sep=',')
surv = np.asarray(colon[['time', 'status']])
surv = np.concatenate((np.zeros((surv.shape[0], 1)), surv), axis=1)
X = np.asarray(colon[['sex', 'age', 'obstruct', 'perfor', 'adhere', 'nodes', 'differ', 'extent', 'surg', 'node4', 'etype']])
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

surv = torch.from_numpy(surv)
X = torch.from_numpy(X)
colon


# Fit
parameters = X.shape[1]
theta = torch.normal(0, 0.01, (parameters, 1), dtype=torch.float64, requires_grad=True)
eta = 0.00
lr = 0.01
optimizer = torch.optim.Adam([theta], lr=lr)
for _ in tqdm.tqdm(range(10000)):
    optimizer.zero_grad()
    linpred = torch.mm(X, theta)
    logL = -loglikelihood(surv, linpred) 
    logL.backward()
    optimizer.step()


    
# Likelihood:
def loglikelihood(surv, linpred, dtype=torch.DoubleTensor):
    '''
    '''
    event_times = surv[surv[:, -1] == 1, 1][:, None]
    risk_set = ((surv[:, 1] >= event_times) * (surv[:, 0] < event_times)).type(dtype)
    return((torch.sum(linpred[surv[:, -1] == 1]) - torch.sum(torch.log(torch.mm(risk_set, torch.exp(linpred))))))


theta= torch.normal(0, 1, (parameters, 1), dtype=torch.float64, requires_grad=False)




linpred = torch.mm(X, theta)


logL = -loglikelihood(surv, linpred) 


tsamples=1776
ctsamples=876
csamples=20
hsamples=200

# approx - 
ll = []
for ii in range(1000):
    idx = np.sort(np.concatenate((
        np.random.choice(np.where((surv[:, -1] ==1).numpy())[0], csamples, replace=False),
        np.random.choice(np.where((surv[:, -1] ==0).numpy())[0], hsamples, replace=False)
        )))
    LINPRED = torch.clone(linpred)
    SURV = torch.clone(surv)
    LINPRED = LINPRED[idx]
    SURV = SURV[idx]

    event_times = SURV[SURV[:, -1] == 1, 1][:, None]
    risk_set = ((SURV[:, 1] >= event_times) * (SURV[:, 0] < event_times)).type(dtype)
    a = torch.sum(LINPRED[SURV[:, -1] == 1]) * (ctsamples/csamples)
    b = torch.sum(torch.log(torch.mm(risk_set, torch.exp(LINPRED)) * (tsamples/hsamples))) * (ctsamples/csamples)

    ll.append((b-a).numpy())

np.mean(ll)



(csamples/hsamples)/(876/900)

(876/900)/(csamples/hsamples)
