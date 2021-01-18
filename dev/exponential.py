'''

'''

# Importing modules
import os
import sys
import tqdm
import torch
import numpy as np

sys.path.append('/Users/alexwjung/Google Drive/projects/tensorcox/src/')

from tensorcox import  TensorCox

def T(lambda_, pred):
    '''
    Simulating from a exponential distribution. (U - Unif(0, 1))
    T = -log(U) / (lambda * exp(pred)) ||| (U - Unif(0, 1))

    args:
        lambda_(int) > 0                exponentiall rate parameter
        pred_ (np.arr((n, 1)))          predictor
    '''
    return(-log(np.random.uniform(0, 1, (pred.shape[0],)))/(lambda_*np.exp(pred)))



#
n = 10000
p = 100
c = 0



X = np.random.binomial(1, 0.25, (n, p))
X

theta = np.random.normal(0, 1, (p, 1))

pred = np.matmu








#
