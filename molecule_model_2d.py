import numpy as np
import scipy.special
from pf_model_base import MoleculePF


class MoleculeBootstrapPF2D(MoleculePF):
    def __init__(self, init_mean, init_cov, diffusion_coef, dt, box, T, pixel_width, multinom_samples):
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.diffusion_coef = diffusion_coef
        self.dt = dt
        self.box = box  # shape (2, 2): [[xmin, xmax], [ymin, ymax]]
        self.T = T
        self.data = None
        self.pixel_width = pixel_width
        self.multinom_samples = multinom_samples

    def reflect(self, x):
        x_reflected = np.copy(x)
        for i in range(x.shape[1]):
            lower, upper = self.box[i]
            x_reflected[:, i] = np.where(x_reflected[:, i] < lower,
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
        pixel_edges = np.arange(self.box[0, 0], self.box[0, 1] + self.pixel_width, self.pixel_width)
        pixel_centers = 0.5 * (pixel_edges[:-1] + pixel_edges[1:])
        mu, scale = x
        denom = (scale)**2

        # Construct Gaussian profile
        gauss = (1. / (np.sqrt(2 * np.pi * denom)) *
                np.exp(-0.5 * ((pixel_centers - mu) ** 2) / denom))
        gauss = gauss / sum(gauss[:-1])
        return gauss
    
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