from __future__ import division
import numpy as np
import numpy as np
import numpy as np
from scipy.stats import multivariate_normal
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

class dicrete:
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

    def echantillonne_avec_clusters(self, n_samples):
        """
        Génère des échantillons à partir du mélange de gaussiennes, en retournant également les clusters.

        :param n_samples: Le nombre d'échantillons à générer.
        :return: Un tuple contenant un tableau d'échantillons et un tableau des indices de clusters.
        """
        n_gaussians = len(self.poids)
        assert np.isclose(sum(self.poids), 1), "La somme des poids doit être égale à 1"

        echantillons = np.zeros((n_samples, 2))
        clusters = np.zeros(n_samples, dtype=int)

        cumulative_sum = np.cumsum(self.poids)
        cumulative_sum[-1] = 1  # Assure que la somme est exactement 1
        batch_limits = np.searchsorted(cumulative_sum, np.random.rand(n_samples))

        for i in range(n_gaussians):
            indices = (batch_limits == i)
            n_gaussian_samples = np.sum(indices)
            echantillons[indices] = np.random.multivariate_normal(self.Y_star[i], self.covariances[i], n_gaussian_samples)
            clusters[indices] = i

        return echantillons, clusters
    
    
    def density(self, x):
        densities = [self.poids[i] * multivariate_normal.pdf(x, mean=self.Y_star[i], cov=self.covariances[i]) 
                  for i in range(len(self.poids))]
        return np.sum(densities, axis=0)

    def gmm_density_heatmap(self):
        """
        Affiche une carte de chaleur représentant la densité du mélange de gaussiennes.
        """
        x_grid, y_grid = np.meshgrid(np.linspace(-5, 15, 50), np.linspace(-5, 15, 50))
        z_grid = np.empty(x_grid.shape)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                x = np.array([x_grid[i, j], y_grid[i, j]])
                z_grid[i, j] = self.density(x)
        plt.contourf(x_grid, y_grid, z_grid, levels=100)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')

    def gmm_score_plot(self, data):
        # Vous devez définir la fonction 'gaussian_mixture_score' pour calculer le score
        # scores_vec = gaussian_mixture_score(data, self.Y_star, self.covariances, self.poids)

        plt.scatter(data[:,0], data[:,1], s = 1, alpha=0.5)
        # plt.quiver(data[:,0], data[:,1], scores_vec[:,0], scores_vec[:,1], 
        #            np.linalg.norm(scores_vec, axis=1), color='red', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gaussian discrete star')
        plt.show()

def gradient_log_start(X, poids, Y_star, covariances):
    """
    Calculate the gradient of the log-likelihood of a Gaussian mixture model.

    :param X: An array of points where the gradient is calculated (numpy array of points).
    :param poids: The weights of each Gaussian in the mixture.
    :param Y_star: The means of each Gaussian in the mixture.
    :param covariances: The covariance matrices of each Gaussian in the mixture.
    :return: The gradient of the log-likelihood of the Gaussian mixture at each point in X.
    """
    n_gaussians = len(poids)
    n_samples = X.shape[0]
    gradients = np.zeros((n_samples, X.shape[1]))
    
    for i in range(n_gaussians):
        diff = X - Y_star[i]
        inv_cov = np.linalg.inv(covariances[i])
        densities = multivariate_normal.pdf(X, mean=Y_star[i], cov=covariances[i])
        grad_gaussiennes = -np.dot(diff, inv_cov) * densities[:, np.newaxis]
        gradients += poids[i] * grad_gaussiennes

    sum_densities = np.sum([poids[i] * multivariate_normal.pdf(X, mean=Y_star[i], cov=covariances[i]) 
                            for i in range(n_gaussians)], axis=0)
    gradients /= sum_densities[:, np.newaxis]

    return gradients
    

def plot_gaussian_mixture_scores(data, scores_vec):
    """
    Plots the scores (gradients) of a Gaussian mixture model for a given dataset.

    :param data: The dataset (numpy array).
    :param scores_vec: The computed scores (gradients) for each data point.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the scores for the dataset
    ax[0].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    ax[0].quiver(data[:, 0], data[:, 1], scores_vec[:, 0], scores_vec[:, 1],
                 np.linalg.norm(scores_vec, axis=1), color='red', alpha=0.5)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Gaussian mixture score for dataset')

    # Plot the score over a grid
    x_grid = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
    y_grid = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    scores_grid = np.array([gradient_log_start(point, poids, Y_star, covariances) for point in grid])
    ax[1].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
    ax[1].quiver(grid[:, 0], grid[:, 1], scores_grid[:, 0], scores_grid[:, 1],
                 np.linalg.norm(scores_grid, axis=1), color='red', alpha=0.5)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Gaussian mixture score over grid')

    fig.suptitle('Gaussian Mixture Model Score')
    fig.tight_layout()
    plt.show()


