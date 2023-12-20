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

def multivariate_gaussian_density(x, mu, sigma):
    """
    Computes the multivariate gaussian density function at x with mean mu and covariance sigma.
    ----------
    Parameters:
    x : numpy array of shape (n, 2)
        The point at which to evaluate the density function.
    mu : numpy array of shape (2,)
        The mean of the gaussian distribution.
    sigma : numpy array of shape (2, 2)
        The covariance matrix of the gaussian distribution.
    """
    if x.ndim == 1:
        return (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))) * np.exp(-0.5 * np.dot(np.dot((x - mu), 
                                                                                             np.linalg.inv(sigma)), (x - mu).T))
    else:
        density = np.zeros(x.shape[0])
        for i in range(len(x)):
            density[i] = (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))) * np.exp(-0.5 * np.dot(np.dot((x[i] - mu), 
                                                                                                         np.linalg.inv(sigma)), 
                                                                                                         (x[i] - mu).T))
        return density


def gmm_density(x, mu, sigmas, alphas):
    """
    Computes the unnormalized density of a Gaussian mixture model.
    ----------
    Parameters:
    x: Points to evaluate the density at. 
    mu: Means of the Gaussian components.
    var: Covariances of the Gaussian components.
    alphas: Mixing proportions of the Gaussian components.
    """
    density = 0
    for i in range(len(mu)):
        density += alphas[i] * multivariate_gaussian_density(x, mu[i], sigmas[i])
    return density

def gmm_log_density(x, mu, sigmas, alphas):
    """
    Computes the unnormalized log density of a Gaussian mixture model.
    ----------
    Parameters:
    x: Points to evaluate the density at. 
    mu: Means of the Gaussian components.
    var: Covariances of the Gaussian components.
    alphas: Mixing proportions of the Gaussian components.
    """
    return np.log(gmm_density(x, mu, sigmas, alphas))


def banana_density(x, mu, sigma, b=0.5):
    """
    Computes the unnormalized log density of the banana-shaped distribution.
    Parameters
    ----------
    x : The point at which to evaluate the density function.
    mu : The mean of the distribution.
    sigma : The covariance matrix of the distribution.
    b : The parameter of the distribution.
    """
    if x.ndim == 1:
        x_transformed = np.copy(x)
        x_transformed[1] = x[1] + b * (x[0]**2 - sigma[0,0])
        return multivariate_gaussian_density(x_transformed, mu, sigma)
    x_transformed = np.copy(x)
    x_transformed[:,1] = x[:,1] + b * (x[:,0]**2 - sigma[0,0])
    return multivariate_gaussian_density(x_transformed, mu, sigma)


def log_density_banana(x, mu, sigma, b=0.5):
    """
    Computes the unnormalized log density of the banana-shaped distribution.
    Parameters
    ----------
    x : The point at which to evaluate the density function.
    mu : The mean of the distribution.
    sigma : The covariance matrix of the distribution.
    b : The parameter of the distribution.
    """
    if x.ndim == 1:
        x_transformed = np.copy(x)
        x_transformed[1] = x[1] + b * (x[0]**2 - sigma[0,0])
        return np.log(multivariate_gaussian_density(x_transformed, mu, sigma))
    x_transformed = np.copy(x)
    x_transformed[:,1] = x[:,1] + b * (x[:,0]**2 - sigma[0,0])
    return np.log(multivariate_gaussian_density(x_transformed, mu, sigma))

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
