# **Bayesian Cox Regression for Population-scale Inference in Electronic Health Records**.

## **Paper**
[arXiv](https://arxiv.org/abs/2106.10057)

## **Description**
The Cox model is an indispensable tool for time-to-event analysis, particularly in biomedical research. However, medicine is undergoing a profound transformation, generating data at an unprecedented scale, which opens new frontiers to study and understand diseases. With the wealth of data collected, new challenges for statistical inference arise, as datasets are often high dimensional, exhibit an increasing number of measurements at irregularly spaced time points, and are simply too large to fit in memory. Many current implementations for time-to-event analysis are ill-suited for these problems as inference is computationally demanding and requires access to the full data at once.
Here we propose a Bayesian version for the counting process representation of Cox's partial likelihood for efficient inference on large-scale datasets with millions of data points and thousands of time-dependent covariates. Through the combination of stochastic variational inference and a reweighting of the log-likelihood, we obtain an approximation for the posterior distribution that factorizes over subsamples of the data, enabling the analysis in big data settings.
Crucially, the method produces viable uncertainty estimates for large-scale and high-dimensional datasets.
We show the utility of our method through a simulation study and an application to myocardial infarction in the UK Biobank.

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
- Heart investigates mortality in patients from the Stanford heart transplant program [@crowley_covariance_1977].
- PBCseq are follow-up laboratory data from the Mayo clinical trial on primary biliary cholangitis and D-penicillamine treatment [@incollectiontherneau_cox_2000].  

The purpose of these examples is to evaluate the performance of our proposed method in comparison to the frequentist Cox model on real world applications.

Furthermore, we used the UK Biobank [@sudlow_uk_2015] to study the association of standard risk factors and comorbidities, taken from the electronic health records, with the occurrence of myocardial infarction.
The UK Biobank (UKB) is a large-scale biomedical database established in 2006. In total there are 502628 participants, recruited between 2006 and 2010. All participants were between 40-69 years of age at their recruitment date.  For details of the analysis see section 4 of the paper.

We provide an algorithm to efficiently simulate event times with time-varying covariates that resemble disease association studies in electronic health records. In total we run two standard case simulations N >> P and a high-dimensional case P >> N. For details of the simulations see section 3 of the paper.

The original scripts/data/outputs from the analysis are in the following folders:
- The accompanying data can be found in [Data](./data)
- Information on the installed packages, the Author Contributions Checklist, and other relevant documents are found in [Docs](./docs)
- Most of the outputs and raw files, like estimates, tables, figures etc. are found in [Out](./out)
- A dedicated folder with the relevant outputs for the paper can be found in [Paper](./paper)
- The scripts used to run the analysis are found in [Scripts](./scripts)
- A dedicated folder for the analysis of the UKB data with subfolder [output](./ukb/out) and [scripts](./ukb/scripts) can be found in [UKB-Analysis](./ukb)

Notebooks for easy replication of the analysis can be found in [Replication](./replication)  
The figures and tables are listed in the order as they appear in the paper (supplementary materials at the end).  
Some of the additional results in the supplementary materials can be produced by the same notebooks as for the main paper.   

Link to the .ipynb files - link to a specific colab session

- Replication notebooks for the applications are in [Replicate Applications](./replication/application)
    - [Colon](./replication/application/colon.ipynb) - [Colab](https://drive.google.com/file/d/1x_Qr3ex96ZFIKe91k1Z_tIBRcK73UfQ4/view?usp=sharing)
    - [Lung](./replication/application/lung.ipynb) - [Colab](https://drive.google.com/file/d/1eUSzHgtipGXoOVaEUwd3Jj2VDgkjj0Dr/view?usp=sharing)
    - [Heart](./replication/application/heart.ipynb) - [Colab](https://drive.google.com/file/d/16FU7wPTghEZ9mmjUwiCOHpQimuqF4idt/view?usp=sharing)
    - [Nafld](./replication/application/nafld.ipynb) - [Colab](https://drive.google.com/file/d/1cN8BMEHQc_WkZDCfgfAzSj7uE7XYMbWE/view?usp=sharing)
    - [PBCseq](./replication/application/pbcseq.ipynb) - [Colab](https://drive.google.com/file/d/1d_fHJgA3kY7g96IBz6INZO87LNc9T6ym/view?usp=sharing)

- Replication notebooks for the simulations are in [Replicate Simulations](./replication/simulations)
    - [Standard Case 1](./replication/simulation/standard_case1.ipynb) - [Colab](https://drive.google.com/file/d/1zSDf9gLQpszHfkyidZqwMoipQtv4Z9MQ/view?usp=sharing)
    - [Standard Case 2](./replication/simulation/standard_case2.ipynb) - [Colab](https://drive.google.com/file/d/1Mj-hxLaJ5wTcUkGiaEaR0IDkVrbSOHof/view?usp=sharing)
    - [High-dimensional Case](./replication/simulation/highdimensional_case.ipynb) - [Colab](https://drive.google.com/file/d/1-dcQ-PRfmIxDyH3O9JVwzwPLnKDCPPg1/view?usp=sharing)
    - [Resources](./replication/simulation/resources.ipynb) - [Colab](https://drive.google.com/file/d/1WqB5CtoGZbknBK1yt3L2e6_aEeQJg_DY/view?usp=sharing)

- To replicate the tables presented in the paper go to [Replicate Tables](./replication/tables)
    - [Data Example](./replication/tables/data_example.ipynb) - [Colab](https://drive.google.com/file/d/1ibBz6-1qdvD13-sxJDmD6Ux2V5S0BauK/view?usp=sharing)
    - [Likelihood Approximation](./replication/tables/likelihood_approx.ipynb) - [Colab](https://drive.google.com/file/d/17phSXoncVM2A-lBCPZyy7y58e4vqNo_v/view?usp=sharing)
    - [Standard Case 1](./replication/tables/standard_case1_table.ipynb) - [Colab](https://drive.google.com/file/d/1g9waGQ0t1dDp3gQyquHAoOWTn3-nF6tS/view?usp=sharing)
    - [Standard Case 2](./replication/tables/standard_case2_table.ipynb) - [Colab](https://drive.google.com/file/d/1z4lBhkLakOwiRakbOy7dIDgAMG423de6/view?usp=sharing)
    - [High-dimensional Case](./replication/tables/highdimensional_case_table.ipynb) - [Colab](https://drive.google.com/file/d/1y4-Zlmb-ncS-AtwThsnzFofTMfJw5BHo/view?usp=sharing)
    - [Likelihood Approximation - large P](./replication/tables/likelihood_approx_additional1.ipynb) - [Colab](https://drive.google.com/file/d/142OipPL3aadxomaOwvoui-iV-IuLhflr/view?usp=sharing)
    - [Likelihood Approximation - predictor](./replication/tables/likelihood_approx_additional2.ipynb) - [Colab](https://drive.google.com/file/d/1u0xA6s_HOGu94amigZOZnP39RNq_pypn/view?usp=sharing)

- To replicate the figures presented in the paper go to [Replicate Figures](./replication/figures)
    - [Schematic](./replication/figures/schematic.ipynb) - [Colab](https://drive.google.com/file/d/1h3Yobtfwi6KyfUesWCeAPW-traP9Bi9r/view?usp=sharing)
    - [Likelihood Approximation](./replication/figures/likelihood_approximation.ipynb) - [Colab](https://drive.google.com/file/d/1BcOKb-1ywakrp-f1AD9td-CA5tMRZ5e2/view?usp=sharing)
    - [High-dimensional](./replication/figures/hd.ipynb) - [Colab](https://drive.google.com/file/d/1KUG1B5gnJioaXEvaPYYoebgEkzu0MrIM/view?usp=sharing)
    - [Resource Comparison](./replication/figures/resource.ipynb) - [Colab](https://drive.google.com/file/d/1wvjgoTnEipaktyq6nw7JJQKFrgX6UNyP/view?usp=sharing)
    - [Forest Plot](./replication/figures/forest_plot.ipynb) - [Colab](https://drive.google.com/file/d/1XGsx5Hp_cuHQh7iW_7-ss549_WndFnUD/view?usp=sharing)
    - [Additional Predictor](./replication/figures/predictor.ipynb) - [Colab](https://drive.google.com/file/d/12fWxgz5c4KgCWf8yNKr9-uislz3BXYN4/view?usp=sharing)
    - [Baseline Hazard](./replication/figures/baseline_hazard.ipynb) - [Colab](https://drive.google.com/file/d/1V6PwT1miY8vrrNllJH6Rg5qFZMq7CA9z/view?usp=sharing)

- To replicate a similar analysis as in the UKB go to [Replicate Fake UKB](./replication/ukb)
    - [Fake Data Generation](./replication/ukb/00_fakedata.ipynb) - [Colab](https://drive.google.com/file/d/175xgLqnI4jVHda3sg38iuNLEvO2DfShJ/view?usp=sharing)
    - [Analysis](./replication/ukb/01_fakeanalysis.ipynb) - [Colab](https://drive.google.com/file/d/1zyRTxI4ptVGolbzfrMf69KTBVp7fl4-U/view?usp=sharing)


### Expected run-time
Approximate time needed to reproduce the analyses on a standard desktop machine:
1-8 hours

### Additional information
- Rerunning all the simulations on a single desktop machine will take a considered amount of time. We therefore provide individual simulation runs (chosen by demand) that can be checked/compared to the results provided on https://github.com/alexwjung/ProbCox.

- The simulation results for the high-dimensional case can suffer from numerical instabilities, this happens for the the particular prior specification of student(nu=1, s=0.001). With s > 0.01 we find the result to stabilize much better, however, there is also a stronger regularization applied.
Our replication results are not exact, however, differences are marginal and the overall result are the same.

- The fake simulation for the UKB data needs to write ~2GB of data. In the colab notebooks this would need to be written to the google drive.

## **Citing**
@article{jung2021bayesian,
  title={Bayesian Cox Regression for Population-scale Inference in Electronic Health Records},
  author={Jung, Alexander W and Gerstung, Moritz},
  journal={arXiv preprint arXiv:2106.10057},
  year={2021}
}

## **License**
[MIT License](./LICENSE)

## **Acknowledgement**
AWJ and MG are supported by grant NNF17OC0027594 from the Novo Nordisk Foundation.
The data for the UK Biobank was accessed by application 45761.
