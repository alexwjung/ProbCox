# **Bayesian Cox Regression for Population-scale Inference in Electronic Health Records**.

## **Paper**
[AOAS](https://www.e-publications.org/ims/submission/AOAS/user/submissionFile/51956?confirm=ae21e3ba)

## **Description**
The Cox model is an indispensable tool for time-to-event analysis, particularly in biomedical research. However, medicine is undergoing a profound transformation, generating data at an unprecedented scale, which opens new frontiers to study and understand diseases. With the wealth of data collected, new challenges for statistical inference arise, as datasets are often high dimensional, exhibit an increasing number of measurements at irregularly spaced time points, and are simply too large to fit in memory. Many current implementations for time-to-event analysis are ill-suited for these problems as inference is computationally demanding and requires access to the full data at once.
Here we propose a Bayesian version for the counting process representation of Cox's partial likelihood for efficient inference on large-scale datasets with millions of data points and thousands of time-dependent covariates. Through the combination of stochastic variational inference and a reweighting of the log-likelihood, we obtain an approximation for the posterior distribution that factorizes over subsamples of the data, enabling the analysis in big data settings.
Crucially, the method produces viable uncertainty estimates for large-scale and high-dimensional datasets.
We show the utility of our method through a simulation study and an application to myocardial infarction in the UK Biobank.

The code is fully written in pytorch/pyro and numpy.

## **Installation**
```
$ pip install probcox
```
## **Getting started**
The main folder for the accompanying paper as well as all the scripts for replication are found in [Paper](./paper/ProbCox).

The main scripts for the package are found in [ProbCox](./src/probcox).

## **Citing**
@article{jung2021bayesian,
  title={Bayesian Cox Regression for Population-scale Inference in Electronic Health Records},
  author={Jung, Alexander W and Gerstung, Moritz},
  journal={arXiv preprint arXiv:2106.10057},
  year={2021}
}

## **License**
[MIT License](./LICENSE)
