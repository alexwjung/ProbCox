
# Modules
# =======================================================================================================================
import os
import sys
import shutil
import subprocess
import tqdm
import time

from lifelines import CoxTimeVaryingFitter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

np.random.seed(9044)

# data
# =======================================================================================================================
print('I8000_P1600')
dd = pd.read_csv('/nfs/nobackup/gerstung/awj/projects/ProbCox/paper/ProbCox/tmp/I8000_P1600.csv')

surv = np.asarray(dd.iloc[:, :3]).astype(float)
X = np.asarray(dd.iloc[:, 3:]).astype(float)

# Run
# =======================================================================================================================
ctv = CoxTimeVaryingFitter(penalizer=0.1)
t0 = time.time()
ctv.fit(dd, event_col="V3", start_col="V1", stop_col="V2", show_progress=True)
t1 = time.time()
print(t1-t0)

#bsub -n 4 -M 16000 -R "rusage[mem=16000]" './prob.sh'; bsub -n 4 -M 16000 -R "rusage[mem=16000]" './plifelines.sh'