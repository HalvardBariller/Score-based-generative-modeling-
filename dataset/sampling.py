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

class EchantillonneurMelangeGaussien:
    def __init__(self, center_star, size_star, m_start, scale=0.001):
        """
        Initialise l'échantillonneur pour un mélange de gaussiennes.

        :param center_star: Le centre de l'étoile autour de laquelle les gaussiennes sont centrées.
        :param size_star: La taille de l'étoile, influant sur l'écart des moyennes des gaussiennes.
        :param m_start: Le nombre de points (et donc de gaussiennes) dans l'étoile.
        :param scale: Le facteur d'échelle pour les covariances, déterminant la concentration des distributions.
        """
        self.center_star = center_star
        self.size_star = size_star
        self.m_start = m_start
        self.scale = scale
        self.Y_star = self.create_star_points()  # Crée les points en forme d'étoile
        self.poids = self.normalize(np.random.rand(self.m_start, 1))  # Normalise les poids aléatoires
        self.covariances = self.genere_covariances_concentre()  # Génère les covariances concentrées

    def create_star_points(self):
        """
        Génère des points formant une étoile pour les moyennes des gaussiennes.

        :return: Un tableau numpy des coordonnées des points en forme d'étoile.
        """
        angles = np.linspace(0, 2 * np.pi, self.m_start, endpoint=False)
        r = self.size_star * (1 + np.sin(5 * angles))  
        x = self.center_star[0] + r * np.cos(angles)
        y = self.center_star[1] + r * np.sin(angles)
        return np.vstack((x, y)).T

    def normalize(self, a):
        """
        Normalise un tableau pour que sa somme soit égale à 1.

        :param a: Le tableau à normaliser.
        :return: Le tableau normalisé.
        """
        return a / np.sum(a)

    def genere_covariances_concentre(self):
        """
        Génère des matrices de covariance très concentrées pour chaque gaussienne.

        :return: Une liste de matrices de covariance.
        """
        covariances = []
        for _ in range(self.m_start):
            cov = np.array([[self.scale, 0], [0, self.scale]])
            covariances.append(cov)
        return covariances

    def echantillonne(self, n_samples):
        """
        Génère des échantillons à partir du mélange de gaussiennes.

        :param n_samples: Le nombre d'échantillons à générer.
        :return: Un tableau d'échantillons.
        """
        n_gaussians = len(self.poids)
        assert np.isclose(sum(self.poids), 1), "La somme des poids doit être égale à 1"
        choix_gaussienne = np.random.choice(n_gaussians, size=n_samples, p=self.poids.ravel())
        echantillons = np.array([np.random.multivariate_normal(self.Y_star[g], self.covariances[g]) for g in choix_gaussienne])
        return echantillons

