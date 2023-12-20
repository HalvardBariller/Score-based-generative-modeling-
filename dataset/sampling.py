import numpy as np
import numpy as np
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import pylab as pyl
import random
from time import time
import warnings
warnings.filterwarnings('ignore')

def gaussian_sampling(mu, sigma, n_samples = 1000):
    """
    Gaussian Sampling using Box-Muller Transform
    --------------------------------------------
    Parameters:
    mu: Mean of the gaussian distribution (2,)
    sigma: Covariance matrix of the gaussian distribution (2,2)
    n_samples: Number of samples to be generated
    """
    X = []
    Y = []
    for i in range(n_samples):
        # Sampling Theta
        theta = np.random.uniform(0, 2*np.pi)
        # Sampling R
        u = np.random.uniform(0, 1)
        r = np.sqrt(-2 * np.log(1 - u))
        # Calculating x and y
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        X.append(x)
        Y.append(y)
    gaussian_data = np.array([X, Y]).T
    gaussian_data = np.dot(gaussian_data, sigma) + mu
    return gaussian_data


def gaussian_mixture_sampling(mu, sigma, alphas, n_samples = 10000):
    """
    Gaussian Mixture Sampling
    -------------------------
    Parameters:
    mu: list of arrays with shape (2,)
        Means of the gaussian distribution
    sigma: list of arrays with shape (2,2)
        Covariances of the gaussian distribution
    alphas: list of floats
        Cluster weights
    n_samples: Number of samples to be generated
    """
    assert np.sum(alphas) == 1, "Sum of alphas must be 1"
    assert type(mu) == list, "mu must be a list"
    assert type(sigma) == list, "sigma must be a list"
    assert type(alphas) == list, "alphas must be a list"
    assert len(mu) == len(sigma), "mu and sigma must have the same length"
    gaussian_data = np.zeros((n_samples, 2))
    clusters = np.zeros(n_samples)
    batches = [0]
    for i in range(len(mu)-1):
        batches.append(int(n_samples * alphas[i]))
    batches.append(n_samples)
    for i in range(len(mu)):
        batch = batches[i+1] - batches[i]
        gaussian_data[batches[i]:batches[i+1],:] = gaussian_sampling(mu[i], sigma[i], batch)
        clusters[batches[i]:batches[i+1]] = i
    return gaussian_data, clusters


def banana_shaped_sampling(N, mu, sigma, d = 2, b=0.5):
    """
    Returns samples from the banana-shaped distribution.
    Parameters
    ----------
    N : The number of samples to generate.
    mu : The mean of the distribution.
    sigma : The covariance matrix of the distribution.
    d : The dimension of the samples.
    b : The parameter of the distribution.
    """
    # Samples from a gaussian distribution
    X = gaussian_sampling(mu, sigma, N)
    # Transformation
    X[:,1] -= b * (X[:,0]**2 - sigma[0,0])

    return X

