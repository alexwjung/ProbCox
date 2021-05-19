![alt text](./docs/logo.png)

# **Probabilistic Cox Regression for ScalableInference with Applications to ElectronicHealth Records**.


## **Paper**
[arXiv]()

## **Description**
The Cox model has been an indispensable tool for much of applied biomedical research. However, the sector went through a profound transformation, generating data at an unprecedented scale, opening new frontiers in the way we can study and understand diseases. With the wealth of data collect, new challenges for statistical inference arise, as data sets are often high dimensional, exhibit an increasing number of measurements at irregularly spaced time points, and are simply too large to fit in memory. Many current implementations for time-to-event analysis are ill-suited for these problems as inference is computationally intensive and often requires access to the full data at once. We propose a Bayesian version of Cox's partial likelihood, based on a counting process representation. In combination with a variational objective for inference and a re-weighting of the log-likelihood, we can obtain an approximation for the posterior distribution, that can be factorized over subsamples of the data. Our approach enables the inclusion of time-varying covariates with viable uncertainty estimates for large-scale and high-dimensional data sets. We show the utility of our method through a simulation study and an application to myocardial infarction in the UK Biobank.

The code is fully written in pytorch/pyro and numpy.

## **Installation**
```
$ pip install probcox
```
## **Getting started**
The main folder for the accompanying paper as well as all the scripts for replication are found in [Paper](./paper/ProbCox).

Some quick jupyter notebooks (Google Colab) to try out the module can be found in [Examples](./examples).
- for a Colab link for a small scale simualtion open [Colab](https://colab.research.google.com/drive/1QiCWAAwFDey2LBshXzwBhn5sGeORzYlF?usp=sharing).
- for a Colab link for a high-dimensional simualtion open [Colab](https://colab.research.google.com/drive/1XAGdms1hWoINLxeThyhD7V0AXY8b-Ixo?usp=sharing).
- for a Colab link for a data example on the larynx data [R - KMsurv](https://cran.r-project.org/web/packages/KMsurv/KMsurv.pdf)  open [Colab](https://colab.research.google.com/drive/12TNil6y4Cyxb7hI6WVLdcDgAw9WA5QBl?usp=sharing).

The main scripts are found in [ProbCox](./src/probcox).

## **Citing**
BIBTEX

## **License**
[MIT License](./LICENSE)
