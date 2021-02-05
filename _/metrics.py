import torch
import pandas as pd
import numpy as np

def concordance(surv, predictions, return_pairs=False):
    # small + offset for censored data
    surv[surv[:, -1]==1, 1] = surv[surv[:, -1]==1, 1] - 0.0000001
    event_times = surv[surv[:, -1]==1, 1] 
    event_hazard = predictions[surv[:, -1]==1, 0]
    concordant = 0 
    disconcordant = 0  
    tx = 0 
    ty = 0
    txy = 0
    for ii in range(event_times.shape[0]):
        risk_set = (surv[:, 0] < event_times[ii]) * (event_times[ii] < surv[:, 1])
        txy += np.sum((event_times[ii] == surv[:, 1]) *(event_hazard[ii] == predictions))-1
        ty += np.sum(event_times[ii] == surv[:, 1])-1
        tx += np.sum(event_hazard[ii] == predictions[risk_set])
        concordant += np.sum(predictions[risk_set] < event_hazard[ii])
        disconcordant += np.sum(predictions[risk_set] > event_hazard[ii])

    if return_pairs:
        return(concordant, disconcordant, txy, tx, ty)
    else: 
        return(((concordant-disconcordant) / (concordant+disconcordant+tx)+1)/2)
    
def RMSE(x, x_hat):
    return(np.sqrt(np.mean((x - x_hat)**2)))

