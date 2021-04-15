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

import warnings
warnings.filterwarnings("ignore")

np.random.seed(567)
torch.manual_seed(234)
# -----------------------------------------------------------------------------------------------------------------------------

sys.path.append('./ProbCox/')
import probcox as pcox #import CoxPartialLikelihood
importlib.reload(pcox)

# -----------------------------------------------------------------------------------------------------------------------------
# importing csv data
nafld = pd.read_csv('./data/real/nafld1.csv', sep=',')
surv = np.asarray(nafld[['futime', 'status']])
surv = np.concatenate((np.zeros((surv.shape[0], 1)), surv), axis=1)
X = np.asarray(nafld.iloc[:, 3:])


surv = torch.from_numpy(surv).type(torch.FloatTensor)
X = torch.from_numpy(X).type(torch.FloatTensor)

total_obs = surv.shape[0]
batch_size = 512
total_events = torch.sum(surv[:, -1] == 1)
sampling_proportion = [total_obs, batch_size, total_events, None]

# -----------------------------------------------------------------------------------------------------------------------------

run = True
eta = 1.0
while run:
    run = False
    pyro.clear_param_store()
    m = pcox.PCox(sampling_proportion=sampling_proportion)
    m.initialize(eta=eta, num_particles=10)
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

a = np.round(out['theta'][1].detach().numpy()[:, 0], 5)
b = np.round(torch.diag(pyro.get_param_store()['AutoMultivariateNormal.scale_tril']).detach().numpy(), 5)
c =np.sign(out['theta'][0].detach().numpy()) == np.sign(out['theta'][2].detach().numpy())
for ii in range(X.shape[1]):
    if c[ii]:
        sig = '*'
    else:
        sig = ''
    print(str(a[ii]) + sig + ', (' + str(b[ii]) + ')')

ci = pcox.concordance(surv.detach(), torch.mm(X, out['theta'][1].detach())).numpy()[None]
print(ci)


np.savetxt('./output/real/nafld1/concordance.txt', ci)
np.savetxt('./output/real/nafld1/se.txt', torch.diag(pyro.get_param_store()['AutoMultivariateNormal.scale_tril']).detach().numpy())

with open('./output/real/nafld1/theta_lower.txt', 'a') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][0].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
with open('./output/real/nafld1/theta.txt', 'a') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][1].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')
with open('./output/real/nafld1/theta_upper.txt', 'w') as write_out:
    write_out.write(''.join([str(ii) + '; ' for ii in out['theta'][2].detach().numpy()[:, 0].tolist()]))
    write_out.write('\n')



import numpy as np
x = np.random.normal(0, 1, (10,))
x
y = -3
x[-5:y]
y = None
x[-5:None]
