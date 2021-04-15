import os
import sys
import tqdm
import importlib
import torch
import numpy as np

dtype = torch.FloatTensor

# custom functions
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

# TVC
#-----------------------------------------------------------------------------------------------------------------------------

class TVC():
    def __init__(self, theta=None, P_binary=None, P_continuous=None,  p_binomial=0.2, sigma_normal=1, dtype=dtype):
        self.theta = theta
        self.P_binary = P_binary
        self.P_continuous = P_continuous
        self.dtype = dtype
        self.p_binomial = p_binomial
        self.sigma_normal = sigma_normal
        super().__init__()

    def make_lambda0(self, scale):
        tsum = 0
        t_lambda = [0]
        psum = 0
        basehaz = [10e-5]
        while tsum < 30000:
            t_lambda.append(np.minimum(30000, t_lambda[-1] + np.round(np.random.gamma(4, 200)).astype(int)))
            if t_lambda[-1] <= 15000:
                basehaz.append(basehaz[-1] + np.random.gamma(2, 1))
            elif np.logical_and(t_lambda[-1] > 15000, t_lambda[-1] <= 25000):
                basehaz.append(basehaz[-1] + np.random.gamma(1, 10))
            else:
                basehaz.append(basehaz[-1] - np.random.gamma(1, 5))
            if basehaz[-1] <= 0:
                basehaz[-1] = 10
            psum += basehaz[-2] * (t_lambda[-1] - t_lambda[-2])
            tsum = t_lambda[-1]
        t_lambda = np.asarray(t_lambda)
        basehaz = np.asarray(basehaz)[:, None]
        basehaz = basehaz / (psum*scale)
        self.t_lambda0, self.lambda0, self.logit_lambda0 = t_lambda, basehaz, logit(basehaz)

    def return_lambda0(self):
        return(self.t_lambda0, self.lambda0)

    def sample(self):
        EOO = np.round(np.random.uniform(1000, 30000)).astype(int)
        # sample covariates:
        tsum = 0
        t_cov = [0]
        X = np.zeros((1, self.P_binary + self.P_continuous))

        while tsum < EOO:
            t_cov.append(np.minimum(EOO, t_cov[-1] + np.round(np.random.gamma(4, 500)).astype(int)))
            X = np.concatenate((X, np.concatenate((np.random.binomial(1, self.p_binomial, (1, self.P_binary)), np.random.normal(0, self.sigma_normal, (1, self.P_continuous))), axis=1)))
            tsum = t_cov[-1]
        t_cov = np.asarray(t_cov)

        # combine with baseline hazard
        basehaz = np.copy(self.logit_lambda0)

        #expand
        basehaz = np.concatenate((basehaz, np.zeros((t_cov.shape[0], 1))))
        X = np.concatenate((np.zeros((self.t_lambda0.shape[0], self.P_binary + self.P_continuous)), X))
        t = np.concatenate((self.t_lambda0, t_cov))
        idx_sort = np.argsort(t)
        t = t[idx_sort]
        basehaz = forward_fill(basehaz[idx_sort])
        X = forward_fill(X[idx_sort])

        idx_r = (t <= EOO)
        t = t[idx_r]
        basehaz = basehaz[idx_r]
        X = X[idx_r]

        # collapse
        X = np.concatenate([np.sum(X[t==ii, :], axis=0)[None, :] for ii in np.unique(t)])
        basehaz = np.concatenate([np.sum(basehaz[t==ii, :], axis=0)[None, :] for ii in np.unique(t)])
        t = np.unique(t)

        # survial data
        H = logisitc(np.matmul(X, self.theta) + basehaz)
        t_diff = (t[1:] - t[:-1])
        event = False

        for ii in range(H.shape[0]-1):
            eval = np.random.uniform(0, 1, (t_diff[ii],)) <= H[ii]
            if np.any(eval):
                event = True
                break
        if event:
            t_event = np.maximum(1, np.where(eval)[0][0])
            X = X[:ii+1]
            t = t[:ii+1]
            t = np.concatenate((t, np.asarray([t[-1] + t_event])))
            time = np.concatenate((t[:-1, None], t[1:, None], np.zeros((X.shape[0], 1))), axis=1)
            time[-1, -1] = 1
        else:
            time = np.concatenate((t[:-1, None], t[1:, None], np.zeros((X.shape[0]-1, 1))), axis=1)
            X = X[:-1]

        # collapse - times by unique covariate shifts
        idx_keep = []
        for ll in range(time.shape[0]-1, 0, -1):
            if np.all(X[ll, :] == X[ll-1, :]):
                idx_keep.append([False])
                time[ll-1, 1] = time[ll, 1]
                time[ll-1, -1] = time[ll, -1]
            else:
                idx_keep.append([True])
        idx_keep.append([True])
        idx_keep = np.asarray(idx_keep)[:, 0]
        idx_keep = np.flip(idx_keep)

        time = time[idx_keep, :]
        X = X[idx_keep, :]

        return(torch.tensor(time).type(self.dtype), torch.tensor(X).type(self.dtype))

    def make_dataset(self, obs, fraction_censored):
        n_censored = np.floor(obs * fraction_censored)
        n_events = np.ceil(obs * (1-fraction_censored))

        surv = torch.zeros((0, 3))
        X = torch.zeros((0, self.P_continuous + self.P_binary))

        n_e = 0
        n_c = 0
        while (n_c + n_e)< obs:
            a, b = self.sample()
            if a[-1, -1] > 0:
                if n_e < n_events:
                    surv = torch.cat((surv, a))
                    X = torch.cat((X, b))
                    n_e += 1
            else:
                if n_c < n_censored:
                    surv = torch.cat((surv, a))
                    X = torch.cat((X, b))
                    n_c += 1
        return(surv, X)
