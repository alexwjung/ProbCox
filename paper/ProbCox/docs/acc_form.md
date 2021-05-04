This form documents the artifacts associated with the article (i.e., the data and code supporting the computational findings) and describes how to reproduce the findings.
# **Probabilistic Cox Regression**
# Abstract
The Cox model has been an indispensable tool for much of applied biomedical research. However, the sector went through a profound transformation, generating data at an unprecedented scale, opening new frontiers in the way we can study and understand diseases. With the wealth of data collect, new challenges for statistical inference arise, as data sets are often high dimensional, exhibit an increasing number of measurements at irregularly spaced time points, and are simply too large to be analyzed at once.
Many current methods for time-to-event analysis are ill-suited for these problems as inference is computationally intensive and often requires access to the full data. We propose a Bayesian version of Cox's partial likelihood, based on a counting process representation to accommodate difficult missing data patterns and time-varying covariates.
Using a variational objective for inference and a re-weighting of the log-likelihood, we can obtain an approximation for the a posteriori distribution, that can be factorized over sub-samples of the data, allowing inference in large-scale settings. In combination with a sparsity-inducing a priori distribution, inference in high-dimensional settings is also possible.Our approach enables the inclusion of time-varying covariates, as well as viable uncertainty estimates for large-scale and high-dimensional data sets.


# Part 1: Data
## Abstract

Publicly Available (taken from R-Survival[@therneau_package_2015]):

**Colon** are data on trials of adjuvant chemotherapy for colon cancer,  [@laurie_surgical_1989]. **Lung** are data extracted from the North Central Cancer Treatment Group on mortality for advanced lung cancer [@loprinzi_prospective_1994].
**NAFLD** is a large population-based study investigating non-alcoholic fatty liver disease (NAFLD) [@allen_nonalcoholic_2018]. **Heart** investigates mortality in patients from the Stanford heart transplant program [@crowley_covariance_1977]

Non-publicly available:
The UK Biobank (UKB) [@sudlow_uk_2015] is a large-scale biomedical database established in 2006. The information collected on individuals includes questionnaires and face-to-face interviews (covering general medical factors, lifestyle, environmental factors, socioeconomics, etc.), physical measurements, blood and urine assays, prescriptions, and genotypes, as well as linkage to their hospital admission records. Further, detailed information, like wearable devices, precise dietary information, is available for smaller subsets. In total there are 502628 participants, recruited between 2006 and 2010. All participants were between 40-69 years of age at their recruitment date.


## Availability
### Publicly available data
Data are available online at: https://github.com/alexwjung/ProbCox

### Non-publicly available data
The UK-Biobank (UKB) data contains sensible information on individuals and cannot be openly shared for privacy considerations.
However, researchers can apply for access to the UKB under:
https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access

The simulation study we present in the paper closely resembles the data structure encountered in electronic health records, similar to the data in the UKB, and can be used as a guidance.

Electronic health records and biobanks contain a vast array of information on individuals that open new possibilities to study disease and are therefore highly important in biomedical research. However, the wealth of information also makes it difficult to obfuscate data in such a way as to preserve important structures in the data.
This inevitably leads to issue for openly distributing the data.


## Description
The data provided can be found in [Data](./data)

### File format(s)

CSV or other plain text.

### Data dictionary
Data file(s) is(are) self-describing (e.g., netCDF files)

# Part 2: Code

## Abstract

We provide a python package (probcox) for the model implemented. This is a scalable version of the standard Cox model fitted via stochastic variational inference. The code is written entirely in pytorch/pyro and numpy.
We provide an algorithm to efficiently simulate survival times with time-varying covariates that resemble disease association studies in electronic health records.
The original scripts for all the analyses are provided. Additionally, there are jupyter notebooks that can be run in a google colab to readily replicate all of the results presented (except for the UKB as we cannot provide the data as mentioned above).

## Description
### Code format(s)

- [ ] Script files
    - [X] R
    - [X] Python
    - [ ] Matlab
    - [ ] Other:
- [ ] Package
    - [X] R
    - [X] Python
    - [ ] MATLAB toolbox
    - [ ] Other:
- [ ] Reproducible report
    - [ ] R Markdown
    - [X] Jupyter notebook
    - [X] Other: Google Colab
- [X] Shell script
- [ ] Other (please specify):

### Supporting software requirements

We provide replicable notebooks that can be used with Google Colab (requirement: Google account).
The notebooks can also be run via standard jupyter notebooks.

#### Version of primary software used

Python version 3.7
R version 4.0.3

#### Libraries and dependencies used by the code
We use conda [https://docs.conda.io/en/latest/] as a package manager and the full list of install packages can be found/replicated via [R environment](./docs/requirements_R.txt) and [Python environment](./docs/requirements_python.txt)

### License

[MIT License](./LICENSE)

### Additional information (optional)

The code for the model can be installed via:

```
$ pip install probcox
```

## Scope

The provided workflow reproduces:

- [X] Any numbers provided in text in the paper
- [X] All tables and figures in the paper
- [ ] Selected tables and figures in the paper, as explained and justified below:

## Workflow
### Instructions
<!--
Describe how to use the materials provided to reproduce analyses in the manuscript. Additional details can be provided in file(s) accompanying the reproducibility materials.
-->


### Expected run-time

Approximate time needed to reproduce the analyses on a standard desktop machine:

- [ ] < 1 minute
- [ ] 1-10 minutes
- [ ] 10-60 minutes
- [X] 1-8 hours
- [ ] > 8 hours
- [ ] Not feasible to run on a desktop machine, as described here:

### Additional information (optional)

- Rerunning all the simulations on a single desktop machine will take a considered amount of time. We therefore provide individual simualtion runs (choosen by demand) that can be checked/compared to the results provided on https://github.com/alexwjung/ProbCox.

# Notes (optional)
The code for the UKB analysis as well as all the scripts for data preperations are also provided [UKB](./ukb).

# References
