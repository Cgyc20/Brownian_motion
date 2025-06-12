import numpy as np
from particles import FeynmanKac
import particles
from scipy.stats import multivariate_normal
import scipy.special
import matplotlib.pyplot as plt
from particles.collectors import Moments
from abc import ABC, abstractmethod


class MoleculePF(ABC, FeynmanKac):

    @abstractmethod
    def M0(self, N):
        pass

    @abstractmethod
    def M(self, t, xp):
        pass

    @abstractmethod
    def logG(self, t, xp, x):
        pass

    @abstractmethod
    def generate_observations(self):
        pass

    @abstractmethod
    def noise_distribution(self, x):
        pass

    @abstractmethod
    def sample_noise(self, x):
        pass

    @abstractmethod
    def evaluate_noise_pdf(self):
        pass

    def gen_path(self):
        path = []
        for t in range(self.T):
            if t == 0:
                sample = self.M0(1)
            else:
                sample = self.M(t=None, xp=path[-1][np.newaxis, :])
            sample = sample[0]  #extract 1D
            path.append(sample)
        return np.array(path)

    def logG(self, t, xp, x):
        if self.data is None:
            raise ValueError("Data has to be set to evaluate likelihood")
        observed = self.data[t]
        N = x.shape[0]
        log_likelihoods = np.zeros(N)

        for i in range(N):
            log_likelihoods[i] = self.evaluate_noise_pdf(x[i], observed)

        return log_likelihoods
    
    def generate_observations(self, path):
        observations = []
        T = path.shape[0]
        for t in range(T):           
            y_t = self.sample_noise(path[t])
            observations.append(y_t)

        return observations


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