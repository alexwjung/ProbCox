#!/bin/sh

source /nfs/nobackup/gerstung/awj/env/conda//bin/activate renv

/nfs/nobackup/gerstung/awj/env/conda/envs/CancerRisk/bin/python3.7 /nfs/nobackup/gerstung/awj/projects/ProbCox/paper/ProbCox/scripts/simulation/highdimensional_case3.py $VAR1
