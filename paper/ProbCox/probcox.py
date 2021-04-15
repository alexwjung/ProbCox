
import os
import sys
import tqdm
import importlib
import config
os.chdir(config.ROOT_DIR)

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

import matplotlib.pyplot as plt

dtype = torch.FloatTensor

# Distributions
# -----------------------------------------------------------------------------------------------------------------------------

class CoxPartialLikelihood(dist.TorchDistribution):
    support = constraints.real
    has_rsample = False

    def __init__(self, pred, sampling_proportion=None, dtype=dtype):
        self.pred = pred
        self.dtype = dtype
        self.sampling_proportion = sampling_proportion
        super(dist.TorchDistribution, self).__init__()

    # HACK: Only used for model initialization.
    def sample(self, sample_shape=torch.Size()):
        return torch.tensor(1.)

    def log_prob(self, surv):
        if self.sampling_proportion:
            censor_ratio = self.sampling_proportion[0]/(self.sampling_proportion[1])
            uncensored_ratio = self.sampling_proportion[2]/self.sampling_proportion[3]
        else:
            censor_ratio = torch.tensor([1])
            uncensored_ratio = torch.tensor([1])
        surv[surv[:, -1] == 1, 1] = surv[surv[:, -1] == 1, 1] - torch.normal(0.00001, 0.000001, (torch.sum(surv[:, -1] == 1),))
        event_times = surv[surv[:, -1] == 1, 1][:, None]
        risk_set = ((surv[:, 1] >= event_times) * (surv[:, 0] < event_times)).type(self.dtype)
        aa = torch.sum(self.pred[surv[:, -1] == 1]) *  uncensored_ratio
        bb = torch.sum(torch.log(torch.mm(risk_set, torch.exp(self.pred)) *  censor_ratio)) *  uncensored_ratio
        return(aa-bb)



# Models
# -----------------------------------------------------------------------------------------------------------------------------

class PCox():
    def __init__(self, predictor=None, guide=None, optimizer=None, loss=None, sampling_proportion=None, dtype=dtype):
        self.predictor = predictor
        self.guide = guide
        self.optimizer = optimizer
        self.loss = loss
        self.sampling_proportion = sampling_proportion
        self.dtype = dtype
        super().__init__()

    def model(self, data):
        # stoachstic update -likelihood adjustment
        if self.sampling_proportion:
            self.sampling_proportion[0] = torch.tensor([self.sampling_proportion[0]])
            self.sampling_proportion[1] = torch.tensor([self.sampling_proportion[1]])
            self.sampling_proportion[2] = torch.tensor([self.sampling_proportion[2]])
            self.sampling_proportion[3] = torch.sum(data[0][:, -1])

        # prior
        if self.predictor:
            pred = self.predictor(data)
        else:

            #theta =  pyro.sample("theta", dist.StudentT(1, loc=0, scale=0.01).expand([data[1].shape[1], 1])).type(self.dtype)
            theta =  pyro.sample("theta", dist.Normal(0, 1).expand([data[1].shape[1], 1])).type(self.dtype)
            pred = torch.mm(data[1], theta)

        # Likelihood
        pyro.sample("obs", CoxPartialLikelihood(pred=pred, sampling_proportion=self.sampling_proportion), obs=data[0])

    def make_guide(self, rank):
        if self.guide:
            self.guide = self.guide
        else:
            if rank:
                self.guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(self.model, rank=rank)
            else:
                self.guide = pyro.infer.autoguide.AutoMultivariateNormal(self.model)

    def make_optimizer(self, eta):
        if self.optimizer:
            pass
        else:

            self.optimizer =pyro.optim.AdagradRMSProp({'eta':eta})
            #self.optimizer = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': 0.0001}, 'gamma': 0.995, 'verbose':False})

    def make_loss(self, num_particles):
        if self.loss:
            pass
        else:
            self.loss = pyro.infer.JitTrace_ELBO(num_particles=num_particles, strict_enumeration_warning=False)

    def return_guide(self):
        return(self.guide)

    def initialize(self, seed=11, num_particles=1, eta=0.5, rank=None):
        pyro.set_rng_seed(seed)
        self.make_guide(rank)
        self.make_optimizer(eta)
        self.make_loss(num_particles)
        self.svi = pyro.infer.SVI(model=self.model, guide=self.guide, optim=self.optimizer, loss=self.loss)

    def infer(self, data):
        ll = self.svi.step(data)
        #self.optimizer.step()
        return(ll)

# Metrics
# -----------------------------------------------------------------------------------------------------------------------------

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
        txy += torch.sum((event_times[ii] == surv[:, 1]) *(event_hazard[ii] == predictions))-1
        ty += torch.sum(event_times[ii] == surv[:, 1])-1
        tx += torch.sum(event_hazard[ii] == predictions[risk_set])
        concordant += torch.sum(predictions[risk_set] < event_hazard[ii])
        disconcordant += torch.sum(predictions[risk_set] > event_hazard[ii])

    if return_pairs:
        return(concordant, disconcordant, txy, tx, ty)
    else:
        return(((concordant-disconcordant) / (concordant+disconcordant+tx)+1)/2)
