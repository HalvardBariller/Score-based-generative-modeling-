import matplotlib.pyplot as plt
import numpy as np

from dataset import densities, sampling



def langevin_trajectory_gmm(x_trajectory, *args):
    """
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    mu, sigma, alphas = args
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = densities.gmm_density(x, mu, sigma, alphas)
    ax.contourf(x_grid, y_grid, z_grid, levels=100, zorder=0)
    fig.colorbar(ax.contourf(x_grid, y_grid, z_grid, levels=100), ax=ax)
    ax.scatter(x_trajectory[:,0], x_trajectory[:,1], s = 2, alpha=1, zorder=1, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectory of the Langevin diffusion')
    plt.show()


def langevin_trajectory_banana(x_trajectory, *args):
    """
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    mu, sigma, b = args
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = densities.banana_density(x, mu, sigma, b)
    ax.contourf(x_grid, y_grid, z_grid, levels=100, zorder=0)
    fig.colorbar(ax.contourf(x_grid, y_grid, z_grid, levels=100), ax=ax)
    ax.scatter(x_trajectory[:,0], x_trajectory[:,1], s = 2, alpha=1, zorder=1, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Trajectory of the Langevin diffusion')
    plt.show()



def langevin_sampling_gmm(x_samples, *args, annealed=False):
    """
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    mu, sigma, alphas = args
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = densities.gmm_density(x, mu, sigma, alphas)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].contourf(x_grid, y_grid, z_grid, levels=100, zorder=0, cmap='RdBu_r')
    fig.colorbar(ax[0].contourf(x_grid, y_grid, z_grid, levels=100, cmap='RdBu_r'), ax=ax[0])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Unnormalized Gaussian mixture density')
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].hist2d(x_samples[:, 0], x_samples[:, 1], cmap='RdBu_r', rasterized=False, 
                 bins=150, density=True, range=[[-5, 15], [-5, 15]])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Empirical density')
    if annealed:
        fig.suptitle('Annealed Langevin Dynamics for Gaussian mixture distribution', fontsize=20)
    else:
        fig.suptitle('Langevin Dynamics for Gaussian mixture distribution', fontsize=20)
    fig.tight_layout()
    plt.show()


def langevin_sampling_banana(x_samples, *args):
    """
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    mu, sigma, b = args
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = densities.banana_density(x, mu, sigma, b)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].contourf(x_grid, y_grid, z_grid, levels=100, zorder=0, cmap='RdBu_r')
    fig.colorbar(ax[0].contourf(x_grid, y_grid, z_grid, levels=100), ax=ax[0])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Unnormalized banana-shaped density')
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].hist2d(x_samples[:, 0], x_samples[:, 1], cmap='RdBu_r', rasterized=False, 
                 bins=150, density=True, range=[[-5, 5], [-5, 5]])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Empirical density')
    fig.suptitle('Langevin Dynamics for Banana distribution', fontsize=20)
    fig.tight_layout()
    plt.show()



def annealed_langevin_sampling_comparison(x_samples, annealed_samples, *args):
    """
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    mu, sigma, alphas = args
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = densities.gmm_density(x, mu, sigma, alphas)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].contourf(x_grid, y_grid, z_grid, levels=100, zorder=0, cmap='RdBu_r')
    # fig.colorbar(ax[0].contour(x_grid, y_grid, z_grid, levels=100, cmap='RdBu_r'), ax=ax[0])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Unnormalized Gaussian mixture density')
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].hist2d(x_samples[:, 0], x_samples[:, 1], cmap='RdBu_r', rasterized=False, 
                 bins=150, density=True, range=[[-5, 15], [-5, 15]])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Empirical density with Langevin Dynamics')
    ax[2].set_aspect('equal', adjustable='box')
    ax[2].hist2d(annealed_samples[:, 0], annealed_samples[:, 1], cmap='RdBu_r', rasterized=False, 
                 bins=150, density=True, range=[[-5, 15], [-5, 15]])
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title('Empirical density with Annealed Langevin Dynamics')

    
    fig.suptitle('Comparison of Langevin Dynamics for Gaussian mixture sampling', fontsize=20)
    fig.tight_layout()
    plt.show()



def langevin_sampling_star(x_samples, star_sampler, annealed=False):
    """
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    x_grid, y_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = star_sampler.density(x)
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].contourf(x_grid, y_grid, z_grid, levels=100, zorder=0, cmap='RdBu_r')
    fig.colorbar(ax[0].contourf(x_grid, y_grid, z_grid, levels=100, cmap='RdBu_r'), ax=ax[0])
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Unnormalized Gaussian mixture density')
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].hist2d(x_samples[:, 0], x_samples[:, 1], cmap='RdBu_r', rasterized=False, 
                 bins=150, density=True, range=[[-3, 3], [-3, 3]])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Empirical density')
    if annealed:
        fig.suptitle('Annealed Langevin Dynamics for Gaussian mixture distribution', fontsize=20)
    else:
        fig.suptitle('Langevin Dynamics for Gaussian mixture distribution', fontsize=20)
    fig.tight_layout()
    plt.show()