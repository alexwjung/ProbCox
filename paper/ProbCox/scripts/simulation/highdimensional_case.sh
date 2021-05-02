#!/bin/sh

source /nfs/nobackup/gerstung/awj/env/conda//bin/activate renv

/nfs/nobackup/gerstung/awj/env/conda/envs/venv/bin/python3.7 /nfs/nobackup/gerstung/awj/projects/ProbCox/scripts/simulation/highdimensional_case.py $VAR1
