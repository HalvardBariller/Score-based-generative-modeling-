# Score-matching-project
Project on Generative Modeling by Score Estimation. This work was conducted as part of the *Probabilistic Graphical Models and Deep Generative Models* class given by Pr. Pierre LATOUCHE and Pr. Pierre-Alexandre Mattei (MVA Master, December 2023).


This work is mainly based on the paper [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600.pdf) by Yang Song and Stefano Ermon (2019), and presents as well Score Matching techniques developed in [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) (Vincent, 2011) and [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) (Hyvärinen, 2005).

## Introduction

The goal of this project is to present the techniques used in score-based generative models. The authors propose a new method for generative modeling based on the estimation of the score function of the data distribution. The score function is estimated by a neural network conditioned by a noise parameter perturbing the original data. The generative process is then performed using an Annealed version of the Monte-Carlo Langevin Dynamics. The authors have shown at the time of publication that their method was competitive with the then state-of-the-art methods on the MNIST, CIFAR10 and CelebA datasets.

## Method

We give some context regarding generative modeling and the relevance of score-based models. We present the method proposed by the authors in the paper and perform toy experiments to illustrate the method and its limitations.

## Experiments

### Score Matching on Toy Distributions

We first perform score matching on toy distributions to illustrate the method. We use the following distributions:
- Gaussian Mixture Model (GMM) with 2 components
- Banana-shaped distribution

A Score Network is trained on each dataset to estimate the score function of the data distribution. We compare the obtained vector field with the true score function, and show the distance between the two with respect to to the $\ell_2$-norm. 


### Langevin Dynamics on Toy Distributions

We then perform Langevin Dynamics on the toy distributions to illustrate the method. We plot the trajectories of the particles in the true vector field.

#### Trajectories of the chains

| GMM | Banana |
| --- | --- |
| ![GMM](/assets/gmm_langevin.gif) | ![Banana](/assets/banana_langevin.gif) |

#### Monte-Carlo Langevin Dynamics 

| GMM | Banana |
| --- | --- |
| ![GMM](/assets/gmm_sampling.png) | ![Banana](/assets/banana_sampling.png) |


We observe in the GMM case that the sampling can't reconcile properly the proportions between the two modes of the distribution. 





