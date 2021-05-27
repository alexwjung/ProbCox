# **Bayesian Cox Regression for Population-scale Inference in Electronic Health Records**.

## **Paper**
[arXiv]()

## **Description**
The Cox model is an indispensable tool for time-to-event analysis, particularly in biomedical research. However, medicine undergoing a profound transformation, generating data at an unprecedented scale, opens new frontiers to study and understand diseases. With the wealth of data collected, new challenges for statistical inference arise, as data sets are often high dimensional, exhibit an increasing number of measurements at irregularly spaced time points, and are simply too large to fit in memory. Many current implementations for time-to-event analysis are ill-suited for these problems as inference is computationally intensive and often requires access to the full data at once. We propose a Bayesian version for the counting process representation of Cox's partial likelihood. Combining a variational objective and a re-weighting of the log-likelihood, we obtain an approximation for the posterior distribution that factorizes over subsamples of the data to enable stochastic variational inference. Our approach enables the inclusion of time-varying covariates with viable uncertainty estimates for large-scale and high-dimensional data sets. We show the utility of our method through a simulation study and an application to myocardial infarction in the UK Biobank.

The code is fully written in pytorch/pyro and numpy.

## **Getting started**

We assume that individuals have an google account and can run the notebooks via google colabs. The replication can similarly be run on a local installation with jupyter notebooks.

The original scripts used to run the analyses, and most of the raw outputs ect. can be found in this folder.
We mainly distinguish between applications (similarly named as the used data) and simulations, where standard cases may be shortened to sim_sc and high dimensional case to sim_hd.

We analyzed a small number of dataset provided in the R-Survival [@therneau_package_2015] package.
These include:
- Colon are data on trials of adjuvant chemotherapy for colon cancer,  [@laurie_surgical_1989].
- Lung are data extracted from the North Central Cancer Treatment Group on mortality for advanced lung cancer [@loprinzi_prospective_1994].
- NAFLD is a large population-based study investigating non-alcoholic fatty liver disease (NAFLD) [@allen_nonalcoholic_2018].
- Heart investigates mortality in patients from the Stanford heart transplant program [@crowley_covariance_1977]
The purpose of these examples is to evaluate the performance of our proposed method in comparison to the frequentist Cox model on real world applications.

Furthermore, we used the UK Biobank [@sudlow_uk_2015] to study the association of standard risk factors and comorbidities, taken from the electronic health records, with the occurrence of myocardial infarction.
The UK Biobank (UKB) is a large-scale biomedical database established in 2006. In total there are 502628 participants, recruited between 2006 and 2010. All participants were between 40-69 years of age at their recruitment date.  For details of the analysis see section 4 of the paper.

We provide an algorithm to efficiently simulate survival times with time-varying covariates that resemble disease association studies in electronic health records. In total we run two standard case simulations N >> P and a high-dimensional case P >> N. for details of the simulations see section 3 of the paper.

The original scripts/data/outputs from the analysis are in the following folders:
- The accompanying data can be found in [Data](./data)
- Information on the installed packages, the Author Contributions Checklist, and other relevant documents are found in [Docs](./docs)
- Most of the outputs and raw files, like estimates, tables, figures etc. are found in [Out](./out)
- A dedicated folder with the relevant outputs for the paper can be found in [Paper](./paper)
- The scripts used to run the analysis are found in [Scripts](./scripts)
- A dedicated folder for the analysis of the UKB data with subfolder [output](./ukb/out) and [scripts](./ukb/scripts) can be found in [UKB-Analysis](./ukb)

Notebooks for easy replication of the analysis can be found in [Replication](./replication)


Link to the .ipynb files - link to a specific colab session

- Replication notebooks for the applications are in [Replicate Applications](./replication/application)
    - [Colon](./replication/application/colon.ipynb) - [Colab](https://colab.research.google.com/drive/1HifKMp2SjKB3NCnNe-vD1EiAf2bQQ7Rp?usp=sharing)
    - [Lung](./replication/application/lung.ipynb) - [Colab](https://colab.research.google.com/drive/1IniSnT1bUINtUnu_owezJ0FWeKyXWgvu?usp=sharing)
    - [Heart](./replication/application/heart.ipynb) - [Colab](https://colab.research.google.com/drive/1bXWSxZA4KvRvxi5xZswDPbdIEaPTrljv?usp=sharing)
    - [Nafld](./replication/application/nafld.ipynb) - [Colab](https://colab.research.google.com/drive/13IJLUfXSqF_3U9dsEBuvo-Vy29r7WLzn?usp=sharing)

- Replication notebooks for the simulations are in [Replicate Simulations](./replication/simulations)
    - [Standard Case 1](./replication/simulation/standard_case1.ipynb) - [Colab](https://colab.research.google.com/drive/1iEoO9hHkgRWzaLhbU9VYhYk6U6V8nffG?usp=sharing)
    - [Standard Case 2](./replication/simulation/standard_case2.ipynb) - [Colab](https://colab.research.google.com/drive/1lIm7d866QtbIxqY6IRhIFrfTECLBWSDn?usp=sharing)
    - [High-dimensional Case](./replication/simulation/highdimensional_case.ipynb) - [Colab](https://colab.research.google.com/drive/1Db9x78fYhhj5yVTalMhKsP6wOm9tArKr?usp=sharing)

- To replicate the tables presented in the paper go to [Replicate Tables](./replication/tables)
    - [Data Example](./replication/simulation/tables/data_example.ipynb) - [Colab](https://colab.research.google.com/drive/1yHM5iDRE0GqTsj7Jpql32PjpNJaopSJX?usp=sharing)
    - [Likelihood Approximation](./replication/simulation/tables/likelihood_approx.ipynb) - [Colab](https://colab.research.google.com/drive/1HJeGSiSX6_plwbgJleY4RjYFa13Gm2O-?usp=sharing)
    - [Likelihood Approximation - large P](./replication/simulation/tables/likelihood_approx_additional1.ipynb) - [Colab](https://colab.research.google.com/drive/1USX1g8PmHkm6Di1WiwAV0u9nJdZ1JtPw?usp=sharing)
    - [Likelihood Approximation- large LP](./replication/simulation/tables/likelihood_approx_additional2.ipynb) - [Colab](https://colab.research.google.com/drive/1Kx2y_E4aSLx6AG0rlQd3pKDJ2F6HR-_f?usp=sharing)
    - [Standard Case 1](./replication/simulation/tables/standard_case1_table.ipynb) - [Colab](https://colab.research.google.com/drive/11XX0E36TUTNnTFhEeW-It7YIm-5vKc4q?usp=sharing)
    - [Standard Case 2](./replication/simulation/tables/standard_case2_table.ipynb) - [Colab](https://colab.research.google.com/drive/13Pt2tMoJAKkgpU-L9KmqWj-tgsgQNBaz?usp=sharing)
    - [High-dimensional Case](./replication/simulation/tables/highdimensional_case_table.ipynb) - [Colab](https://colab.research.google.com/drive/1Uj6lQaivKj7UaEgR-j5feZgXFhXke0R1?usp=sharing)

- To replicate the figures presented in the paper go to [Replicate Figures](./replication/figures)
    - [Schematic](./replication/simulation/figures/schematic.ipynb) - [Colab](https://colab.research.google.com/drive/1Hz1IG6z4fOJBTNEIM6jSnyO6l586P3G1?usp=sharing)
    - [Likelihood Approximation over training](./replication/simulation/figures/likelihood_training.ipynb) - [Colab](https://colab.research.google.com/drive/1kz42UvTAag7XxEWCgMhw6GidP_fuwW4p?usp=sharing)
    - [High-dimensional](./replication/simulation/figures/hd.ipynb) - [Colab](https://colab.research.google.com/drive/1i_NbMRESZTNSHsqRlnRu0GuPA658UT9W?usp=sharing)
    - [Baseline Hazard](./replication/simulation/figures/) - [Colab](https://colab.research.google.com/drive/1PDp2G-ob1tjIlnh03j9TyoH7QlxDuGYM?usp=sharing)

- To replicate a similar analysis as in the UKB go to [Replicate Fake UKB](./replication/ukb)
    - [Fake Data Generation](./replication/ukb/00_fakedata.ipynb) - [Colab](https://colab.research.google.com/drive/1wT4pw2WEk6npzx7lrSaOjo5JUwTEfVXr?usp=sharing)
    - [Analysis](./replication/ukb/01_fakeanalysis.ipynb) - [Colab](https://colab.research.google.com/drive/1dP4TCF12Nx50bgn7GA2YkBNo9fAFbD2M?usp=sharing)


### Expected run-time

Approximate time needed to reproduce the analyses on a standard desktop machine:
1-8 hours

### Additional information
- Rerunning all the simulations on a single desktop machine will take a considered amount of time. We therefore provide individual simulation runs (chosen by demand) that can be checked/compared to the results provided on https://github.com/alexwjung/ProbCox.

- The simulation results for the high-dimensional case can suffer from numerical instabilities, this happens for the the particular prior specification of student(nu=1, s=0.001). With s > 0.01 we find the result to stabilize much better, however, there is also a stronger regularization applied.
Our replication results are not exact, however, differences are marginal and the overall result are the same.

- The fake simulation for the UKB data needs to write ~2GB of data. In the colab notebooks this would need to be written to the google drive.

## **Citing**
BIBTEX

## **License**
[MIT License](./LICENSE)

## **Acknowledgement**
AWJ and MG are supported by grant NNF17OC0027594 from the Novo Nordisk Foundation.
The data for the UK Biobank was accessed by application 45761.
