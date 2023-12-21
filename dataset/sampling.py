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

class Discrete:
    def __init__(self, center_star, size_star, m_start, scale=0.001):
        """
        Initializes the sampler for a Gaussian mixture.

        :param center_star: The center of the star around which the Gaussians are centered.
        :param size_star: The size of the star, influencing the spread of the Gaussians' means.
        :param m_start: The number of points (and thus Gaussians) in the star.
        :param scale: The scale factor for the covariances, determining the concentration of the distributions.
        """
        self.center_star = center_star
        self.size_star = size_star
        self.m_start = m_start
        self.scale = scale
        self.Y_star = self.create_star_points()
        self.weights = self.normalize(np.random.rand(self.m_start, 1))
        self.covariances = self.generate_concentrated_covariances()

    def create_star_points(self):
        """
        Generates star-shaped points for the means of the Gaussians.

        :return: A numpy array of the coordinates of the star-shaped points.
        """

        angles = np.linspace(0, 2 * np.pi, self.m_start, endpoint=False)
        r = self.size_star * (1 + np.sin(5 * angles))  
        x = self.center_star[0] + r * np.cos(angles)
        y = self.center_star[1] + r * np.sin(angles)
        return np.vstack((x, y)).T

    def normalize(self, a):
        """
        Normalizes an array so that its sum equals 1.

        :param a: The array to be normalized.
        :return: The normalized array.
         """

        return a / np.sum(a)

    def generate_concentrated_covariances(self):
        """
        Generates highly concentrated covariance matrices for each Gaussian.

        :return: A list of covariance matrices.
        """

        covariances = []
        for _ in range(self.m_start):
            cov = np.array([[self.scale, 0], [0, self.scale]])
            covariances.append(cov)
        return covariances

    def sample_with_clusters(self, n_samples):
        """
        Generates samples from the Gaussian mixture, also returning the clusters.

        :param n_samples: The number of samples to generate.
        :return: A tuple containing an array of samples and an array of cluster indices.
        """

        n_gaussians = len(self.weights)
        assert np.isclose(sum(self.weights), 1) 

        echantillons = np.zeros((n_samples, 2))
        clusters = np.zeros(n_samples, dtype=int)

        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1  # Assure que la somme est exactement 1
        batch_limits = np.searchsorted(cumulative_sum, np.random.rand(n_samples))

        for i in range(n_gaussians):
            indices = (batch_limits == i)
            n_gaussian_samples = np.sum(indices)
            echantillons[indices] = np.random.multivariate_normal(self.Y_star[i], self.covariances[i], n_gaussian_samples)
            clusters[indices] = i

        return echantillons, clusters
    
    
    def density(self, x):
        densities = [self.weights[i] * multivariate_normal.pdf(x, mean=self.Y_star[i], cov=self.covariances[i]) 
                  for i in range(len(self.weights))]
        return np.sum(densities, axis=0)

    def gmm_density_heatmap(self):

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

    def plot_simple(self, data):

        plt.scatter(data[:,0], data[:,1], s = 1, alpha=0.5)
        # plt.quiver(data[:,0], data[:,1], scores_vec[:,0], scores_vec[:,1], 
        #            np.linalg.norm(scores_vec, axis=1), color='red', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Gaussian discrete star')
        plt.show()

    def gradient_log_star(self, X):
        """
        Calculate the gradient of the log-likelihood of a Gaussian mixture model.

        :param X: An array of points where the gradient is calculated (numpy array of points).
        :return: The gradient of the log-likelihood of the Gaussian mixture at each point in X.
        """
        n_gaussians = len(self.weights)
        n_samples = X.shape[0]
        gradients = np.zeros((n_samples, X.shape[1]))
        
        for i in range(n_gaussians):
            diff = X - self.Y_star[i]
            inv_cov = np.linalg.inv(self.covariances[i])
            densities = multivariate_normal.pdf(X, mean=self.Y_star[i], cov=self.covariances[i])
            grad_gaussiennes = -np.dot(diff, inv_cov) * densities[:, np.newaxis]
            gradients += self.weights[i] * grad_gaussiennes

        sum_densities = np.sum([self.weights[i] * multivariate_normal.pdf(X, mean=self.Y_star[i], cov=self.covariances[i]) 
                                for i in range(n_gaussians)], axis=0)
        gradients /= sum_densities[:, np.newaxis]

        return gradients
    
    def plot_star_scores(self, data):
        """
        Plots the scores (gradients) of a Gaussian mixture model for a given dataset.

        :param data: The dataset (numpy array).
        """
        scores_vec = self.gradient_log_star(data)
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
        scores_grid = self.gradient_log_star(grid)
        ax[1].scatter(data[:, 0], data[:, 1], s=1, alpha=0.5)
        ax[1].quiver(grid[:, 0], grid[:, 1], scores_grid[:, 0], scores_grid[:, 1],
                     np.linalg.norm(scores_grid, axis=1), color='red', alpha=0.5)
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('y')
        ax[1].set_title('Gaussian mixture score over grid')

        # Visualize the score vectors on the grid
        ax[1].quiver(grid[:, 0], grid[:, 1], scores_grid[:, 0], scores_grid[:, 1],
                     np.linalg.norm(scores_grid, axis=1), color='red', alpha=0.5)
        
        fig.suptitle('Gaussian Mixture Model Score Visualization')
        fig.tight_layout()
        plt.show()



