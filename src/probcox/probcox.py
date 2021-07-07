'''

Main file for the Probabilistic Cox regression
- Cox Partial likelihood
- VI model specification


'''

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

import numpy as np
from multiprocessing import Pool

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
            censor_ratio = torch.tensor([self.sampling_proportion[0]/self.sampling_proportion[1]]).type(self.dtype)
            uncensored_ratio = torch.tensor([self.sampling_proportion[2]/self.sampling_proportion[3]]).type(self.dtype)
        else:
            censor_ratio = torch.tensor([1]).type(self.dtype)
            uncensored_ratio = torch.tensor([1]).type(self.dtype)

        # random tie breaking
        surv[surv[:, -1] == 1, 1] = surv[surv[:, -1] == 1, 1] - torch.normal(0.00001, 0.000001, (torch.sum(surv[:, -1] == 1),)).type(self.dtype)
        event_times = surv[surv[:, -1] == 1, 1][:, None]
        risk_set = ((surv[:, 1] >= event_times) * (surv[:, 0] < event_times)).type(self.dtype)
        aa = torch.sum(self.pred[surv[:, -1] == 1]) *  uncensored_ratio
        bb = torch.sum(torch.log(torch.mm(risk_set, torch.exp(self.pred)) * censor_ratio)) *  uncensored_ratio
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
            theta =  pyro.sample("theta", dist.Normal(0, 1).expand([data[1].shape[1], 1])).type(self.dtype)
            pred = torch.mm(data[1], theta)

        # Likelihood
        pyro.sample("obs", CoxPartialLikelihood(pred=pred, sampling_proportion=self.sampling_proportion, dtype=self.dtype), obs=data[0])

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

    def make_loss(self, num_particles):
        if self.loss:
            pass
        else:
            self.loss = pyro.infer.JitTrace_ELBO(num_particles=num_particles, strict_enumeration_warning=False)

    def return_guide(self):
        return(self.guide)

    def return_model(self):
        return(self.model)

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
class metrics():
    def __init__(self, surv, linpred, processes=10):
        self.surv = surv
        self.linpred = linpred
        self.processes = processes
        super().__init__()

    def compute_concordance(self, ii):
        self.surv[self.surv[:, -1]==1, 1] = self.surv[self.surv[:, -1]==1, 1] - 0.0000001
        event_times = self.surv[self.surv[:, -1]==1, 1]
        event_hazard = self.linpred[self.surv[:, -1]==1, 0]
        risk_set = (self.surv[:, 0] < event_times[ii]) * (event_times[ii] < self.surv[:, 1])
        txy = np.sum((event_times[ii] == self.surv[:, 1]) * (event_hazard[ii] == self.linpred[:, 0]))-1
        ty = np.sum(event_times[ii] == self.surv[:, 1])-1
        tx = np.sum(event_hazard[ii] == self.linpred[risk_set])
        concordant = np.sum(self.linpred[risk_set] < event_hazard[ii])
        disconcordant = np.sum(self.linpred[risk_set] > event_hazard[ii])
        return(concordant, disconcordant, txy, tx, ty)

    def concordance(self):
        concordant = 0
        disconcordant = 0
        txy = 0
        tx = 0
        ty = 0
        with Pool(processes=self.processes) as pool:
            for kk in pool.imap_unordered(self.compute_concordance, range(np.sum(self.surv[:, -1]).astype(int))):
                concordant += kk[0]
                disconcordant += kk[1]
                txy += kk[2]
                tx += kk[3]
                ty += kk[4]
        ci = ((concordant-disconcordant) / (concordant+disconcordant+tx)+1)/2
        return(ci)

def KM(times, events, t1=None):
    '''
    '''
    tmax = np.max(times)
    idx = np.argsort(times)
    times = times[idx]
    events = events[idx]
    helpvar = np.arange(times.shape[0], 0, -1)

    idx_cases = events == 1
    times = times[idx_cases]
    events = events[idx_cases]
    helpvar = helpvar[idx_cases]

    events = np.asarray([np.sum(events[jj == times], axis=0) for jj in np.unique(times)])
    helpvar = np.asarray([np.max(helpvar[jj == times], axis=0) for jj in np.unique(times)])
    times = np.unique(times)
    km = np.cumprod((1 - events/helpvar))

    if t1 != None:
        tmax = np.maximum(t1, tmax)

    times = np.concatenate(([0], times, [tmax]))
    km = np.concatenate(([1],  km, [km[-1]]))

    return(times, km)
