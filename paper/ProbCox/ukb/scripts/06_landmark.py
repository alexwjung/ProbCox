'''
landmark prediction of 10 year myocardial risk - at 01/01/2010 - additional
'''

# Modules
# =======================================================================================================================

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

import matplotlib.pyplot as plt

import probcox

import warnings
warnings.filterwarnings("ignore")

dtype = torch.FloatTensor

np.random.seed(87)
torch.manual_seed(34)

ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'

# ICD10 Codes
# =======================================================================================================================

actionable_codes = ["I20 (angina pectoris)", "I21 (acute myocardial infarction)", "I22 (subsequent myocardial infarction)", "I23 (certain current complications following acute myocardial infarction)", "I24 (other acute ischaemic heart diseases)", "I25 (chronic ischaemic heart disease)", "I60 (subarachnoid haemorrhage)", "I61 (intracerebral haemorrhage)", "I62 (other nontraumatic intracranial haemorrhage)", "I63 (cerebral infarction)", "I64 (stroke, not specified as haemorrhage or infarction)", "I65 (occlusion and stenosis of precerebral arteries, not resulting in cerebral infarction)", "I66 (occlusion and stenosis of cerebral arteries, not resulting in cerebral infarction)", "I67 (other cerebrovascular diseases)", "I68 (cerebrovascular disorders in diseases classified elsewhere)", "I69 (sequelae of cerebrovascular disease)", "I46 (cardiac arrest)", "I50 (heart failure)", 'G45 (transient cerebral ischaemic attacks and related syndromes)']

event_codes = ["I21 (acute myocardial infarction)", "I22 (subsequent myocardial infarction)", "I23 (certain current complications following acute myocardial infarction)", "I24 (other acute ischaemic heart diseases)"]

icd10_codes = pd.read_csv(ROOT_DIR + 'projects/ProbCox/data/icd10_codes.csv', header=None)
icd10_codes
icd10_codes.iloc[:, 1] = icd10_codes.iloc[:, 1].apply(lambda x: x[20:])
icd10_code_names = np.asarray(icd10_codes.loc[icd10_codes.iloc[:, 1].apply(lambda x: x not in actionable_codes), 1])
icd10_code_names = icd10_code_names.tolist()
icd10_code_names.remove('E66 (obesity)')
icd10_code_names.remove('I10 (essential (primary) hypertension)')
icd10_code_names = np.asarray(icd10_code_names)
icd10_codes = icd10_codes.groupby(0).first()


# Dataloader Settings:
# =======================================================================================================================

train = glob.glob('/nfs/research1/gerstung/sds/sds-ukb-cancer/projects/ProbCox/data/prepared/train/**/*', recursive=True)
ll = []
for _ in range(len(train)):
    if np.logical_and(len(train[_].split('/')) == 14, train[_].split('/')[-1] != 'removed.txt'):
        ll.append(True)
    else:
        ll.append(False)
train = np.asarray(train)[ll].tolist()
len(train)

valid = glob.glob('/nfs/research1/gerstung/sds/sds-ukb-cancer/projects/ProbCox/data/prepared/valid/**/*', recursive=True)
ll = []
for _ in range(len(valid)):
    if np.logical_and(len(valid[_].split('/')) == 14, valid[_].split('/')[-1] != 'removed.txt'):
        ll.append(True)
    else:
        ll.append(False)
valid = np.asarray(valid)[ll].tolist()

test = glob.glob('/nfs/research1/gerstung/sds/sds-ukb-cancer/projects/ProbCox/data/prepared/test/**/*', recursive=True)
ll = []
for _ in range(len(test)):
    if np.logical_and(len(test[_].split('/')) == 14, test[_].split('/')[-1] != 'removed.txt'):
        ll.append(True)
    else:
        ll.append(False)
test = np.asarray(test)[ll].tolist()

class RandomSampler(Sampler):
    def __init__(self, ids):
        self.ids_len = len(ids)
        self.ids = ids

    def __iter__(self):
        return iter(np.random.choice(self.ids, self.ids_len, replace=False).tolist())

    def __len__(self):
        return self.ids_len

def custom_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) > 0:
        return([torch.cat([item[ii] for item in batch], 0) for ii in range(len(batch[0]))])
    else:
        return(None, None)

class UKB(Dataset):
    def __init__(self, breslow, theta):
        """
        """
        self.breslow = breslow
        self.theta = theta

    def compute_haz(self, a0, tt, dt, start):
        stop = start + 365*10
        idx = np.logical_and(tt >= start, tt <= stop)
        idx[np.maximum(np.min(np.where(idx)[0])-1, 0)] = True
        idx[np.minimum(np.max(np.where(idx)[0])+1, idx.shape[0]-1)] = True
        dt_ = dt[idx]
        a0_ = a0[idx]
        if start > np.min(tt):
            dt_[0] = np.round(tt[idx][1]-start)
        dt_[-1] = np.round(stop - tt[idx][-2])
        return(np.sum(dt_ * a0_))

    def __len__(self):
        return(500)

    def __getitem__(self, ii, dtype=dtype):
        d = torch.load(ii)
        time = d['time'].type(dtype)
        X = d['X'].type(dtype)

        # rescale BMI -> 10kg/m2
        BMI = torch.cat((
        (X[:, 3] < 18.5).type(dtype)[:, None],
        ((X[:, 3] >= 25).type(dtype)  * (X[:, 3] < 30).type(dtype)) [:, None],
        (X[:, 3] >= 30).type(dtype)[:, None]
        ), axis=1)

        # add icd10 obesity
        BMI[:, 2] = np.minimum(X[:, 252] + BMI[:, 2], 1)
        X = np.delete(X, 252, axis=1)

        # rescale LDL mmol/L
        X[:, 4] = (X[:, 4] - 3.516)

        # rescale HDL mmol/L
        X[:, 5] = (X[:, 5] - 1.4)

        # rescale triglyceride 10 mmol/L
        X[:, 6] = (X[:, 6] - 1.482)

        # blood pressure
        b_pressure = torch.cat((
        ((X[:, 8] >= 120).type(dtype)  * (X[:, 8] < 130).type(dtype) * (X[:, 7] < 80).type(dtype))[:, None],
        torch.logical_or(((X[:, 8] >= 130) * (X[:, 8] < 140)),  ((X[:, 7] >= 80) * (X[:, 7] < 90))).type(dtype)[:, None],
        torch.logical_or(((X[:, 8] >= 140)),  ((X[:, 7] >= 90))).type(dtype)[:, None]), axis=1)

        # add icd10 essential primary hypertension
        b_pressure[:, 2] = b_pressure[:, 2] + X[:, 495]
        b_pressure[:, 1] = b_pressure[:, 1] - b_pressure[:, 2]
        b_pressure = np.maximum(0, b_pressure)
        b_pressure = np.minimum(1, b_pressure)
        X = np.delete(X, 494, axis=1)

        # adj X
        X = torch.cat((X[:, 0, None], X[:, 1, None], X[:, 2, None], X[:, 4, None], X[:, 5, None], X[:, 6, None], X[:, 9, None], BMI, b_pressure, X[:, 10:]), axis=1)

        start = torch.tensor([(np.array(['2010-01-01'], dtype='datetime64[D]') - (d['date'] - d['time'][0, 0].numpy().astype(int))).astype(int)])[:, 0]
        if start > torch.max(d['time']):
            res = None
        else:
            idx = torch.logical_and(d['time'][:, 0] < start,  d['time'][:, 1] >= start)
            linpred = torch.mm(X[idx], self.theta).numpy()
            endpoint = torch.minimum(start + 365*10, torch.max(d['time'])).numpy()

            if d['time'][d['time'][:, 1]  <= (start + 365*10)].shape[0] == 0:
                event = np.asarray([0.])
            else:
                event = torch.max(d['time'][d['time'][:, 1]  <= (start + 365*10), -1]).numpy()[None]
            start = start.numpy()
            endpoint = endpoint-start

            risk10 = self.compute_haz(a0=self.breslow['A0'], tt=self.breslow['tt'], dt=self.breslow['dt'], start=start) * np.exp(linpred)
            if risk10.shape[0]==0:
                res = None
            else:
                res = np.concatenate((endpoint[:, None], event[:, None], risk10), axis=1)
                res = [torch.from_numpy(res)]

            return(res)

out = torch.load(ROOT_DIR + 'projects/ProbCox/output/guide')
breslow = torch.load(ROOT_DIR + 'projects/ProbCox/output/Breslow')
UKB_loader = UKB(breslow=breslow, theta=out['theta'][1].detach())

# Predictions
# =======================================================================================================================
dataloader = DataLoader(UKB_loader, batch_size=500, num_workers=8, prefetch_factor=1, persistent_workers=True, sampler=RandomSampler(train), drop_last=False, collate_fn=custom_collate)
res_train = np.zeros((0, 3))
for _, __input__ in tqdm.tqdm(enumerate(dataloader)):
    res_train = np.concatenate((res_train, __input__[0].numpy()))

dataloader = DataLoader(UKB_loader, batch_size=500, num_workers=8, prefetch_factor=1, persistent_workers=True, sampler=RandomSampler(valid), drop_last=False, collate_fn=custom_collate)
res_valid = np.zeros((0, 3))
for _, __input__ in tqdm.tqdm(enumerate(dataloader)):
    res_valid = np.concatenate((res_valid, __input__[0].numpy()))

dataloader = DataLoader(UKB_loader, batch_size=500, num_workers=8, prefetch_factor=1, persistent_workers=True, sampler=RandomSampler(test), drop_last=False, collate_fn=custom_collate)
res_test = np.zeros((0, 3))
for _, __input__ in tqdm.tqdm(enumerate(dataloader)):
    res_test = np.concatenate((res_test, __input__[0].numpy()))

torch.save({'train':res_train,
           'valid': res_valid,
           'test': res_test}, ROOT_DIR + 'projects/ProbCox/output/landmark')
