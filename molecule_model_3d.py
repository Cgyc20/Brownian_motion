import numpy as np
import scipy.special
from pf_model_base import MoleculePF

class MoleculeBootstrapPF3D(MoleculePF):
    def __init__(self, init_mean, init_cov, diffusion_coef, dt, box, T, pixel_width, multinom_samples):
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.diffusion_coef = diffusion_coef
        self.dt = dt
        self.box = box  # shape (3, 3): [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        self.T = T
        self.data = None
        self.pixel_width = pixel_width
        self.multinom_samples = multinom_samples
    
    def reflect(self, x):
        x_reflected = np.copy(x)
        for i in range(x.shape[1]):
            lower, upper = self.box[i]
            x_reflected[:,i] = np.where(x_reflected[:, i] < lower,
                                         2 * lower - x_reflected[:, i],
                                         x_reflected[:, i])
            x_reflected[:, i] = np.where(x_reflected[:, i] > upper,
                                         2 * upper - x_reflected[:, i],
                                         x_reflected[:, i])
            
        return x_reflected

    def M0(self, N):
        samples = np.random.multivariate_normal(self.init_mean, self.init_cov, size=N)
        return self.reflect(samples)
    
    def M(self, t, xp):
        N, d = xp.shape
        noise = np.random.normal(loc=0.0,
                                 scale=np.sqrt(2 * self.diffusion_coef * self.dt),
                                 size=(N, d))
        x_new = xp + noise
        return self.reflect(x_new)
    
    def noise_distribution(self, x):
        """ 
        pixel_edges = np.arange(self.box[0, 0], self.box[0, 1] + self.pixel_width, self.pixel_width)
        pixel_centers = 0.5 * (pixel_edges[:-1] + pixel_edges[1:])
        mu, scale = x
        denom = (scale)**2
        """
        
        mu, scale = x  # mu = (mu_x, mu_y)
        mu_x, mu_y = mu
        if np.isscalar(scale):
            scale_x = scale_y = scale
        else:
            scale_x, scale_y = scale
        
        # Create 2D pixel grid over x and y dimensions
        x_edges = np.arange(self.box[0, 0], self.box[0, 1] + self.pixel_width, self.pixel_width)
        y_edges = np.arange(self.box[1, 0], self.box[1, 1] + self.pixel_width, self.pixel_width)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        X, Y = np.meshgrid(x_centers, y_centers)  # shape (num_y, num_x)

        # 2D Gaussian formula (unnormalized)
        exponent = -((X - mu_x)**2 / (2 * scale_x**2) + (Y - mu_y)**2 / (2 * scale_y**2))
        prob_grid = np.exp(exponent)

        # Normalize to make it a probability distribution
        prob_grid /= prob_grid.sum()