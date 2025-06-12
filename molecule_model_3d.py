import numpy as np
import scipy.special
from pf_model_base import MoleculePF

class MoleculeBootstrapPF3D(MoleculePF):
    def __init__(self, init_mean, init_cov, diffusion_coef, dt, box, T, pixel_width, multinom_samples):
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.diffusion_coef = diffusion_coef
        self.dt = dt
        self.box = box  # shape (3, 2): [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
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
        mu, scale = x
        mu_x, mu_y = mu
        denom = scale**2

        #Pixel edges and centers for x and y
        x_edges = np.arange(self.box[0, 0], self.box[0, 1] + self.pixel_width, self.pixel_width)
        y_edges = np.arange(self.box[1, 0], self.box[1, 1] + self.pixel_width, self.pixel_width)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        # Create 2D grid of pixel centers
        X, Y = np.meshgrid(x_centers, y_centers)  # Shape: (ny, nx)

        # 2D Gaussian formula (unnormalized)
        gauss_2d = (1. / (2 * np.pi * scale * scale)) * np.exp(
            -0.5 * (((X - mu_x) ** 2) / denom + ((Y - mu_y) ** 2) / denom)
        )

        # Normalize
        gauss_2d /= gauss_2d.sum()

        return gauss_2d
    
    def evaluate_noise_pdf(self, x, data):
        # Avoid log(0) issues
        noise_dist = self.noise_distribution(x=x)
        probs = np.clip(noise_dist, 1e-12, 1.0) #TODO do I need this? -- I do get problems without it, set log manually to -infty if zero
        probs = probs / np.sum(probs)  # Renormalize after clipping

        # Multinomial log-PMF
        log_coef = scipy.special.gammaln(self.multinom_samples + 1) - np.sum(scipy.special.gammaln(data + 1))
        log_prob = np.sum(data * np.log(probs))
        log_likelihood = log_coef + log_prob
        return log_likelihood
    
    def sample_noise(self, x):
        gauss = self.noise_distribution(x=x)
        y = np.random.multinomial(self.multinom_samples, gauss)
        return y