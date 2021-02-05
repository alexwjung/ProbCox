import torch
import pandas as pd
import numpy as np

def loglikelihood(surv, linpred, dtype=torch.DoubleTensor):
    '''
    '''
    event_times = surv[surv[:, -1] == 1, 1][:, None]
    risk_set = ((surv[:, 1] >= event_times) * (surv[:, 0] < event_times)).type(dtype)
    return((torch.sum(linpred[surv[:, -1] == 1]) - torch.sum(torch.log(torch.mm(risk_set, torch.exp(linpred))))))

def Fisher(surv, X, linpred,  dtype=torch.DoubleTensor):
    '''
    '''
    with torch.no_grad():
        event_times = surv[surv[:, -1] == 1, 1][:, None]
        risk_set = ((surv[:, 1] >= event_times) * (surv[:, 0] < event_times)).type(dtype)

        W = ((risk_set * torch.exp(linpred).T) / torch.mm(risk_set, torch.exp(linpred)))
        Z = torch.mm(W, X) 
        W = torch.sqrt(W)

        F = torch.from_numpy(np.zeros((X.shape[1], X.shape[1])))
        for ii in range(risk_set.shape[0]):
            F_ = (X - Z[ii, :]) * W[ii, :][:, None]
            F += torch.mm(F_.T, F_)
    return(F)


