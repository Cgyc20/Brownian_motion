import numpy as np
from particles import FeynmanKac
from abc import ABC, abstractmethod


class MoleculePF(ABC, FeynmanKac):

    @abstractmethod
    def M0(self, N):
        pass

    @abstractmethod
    def M(self, t, xp):
        pass

    @abstractmethod
    def noise_distribution(self, x):
        pass

    @abstractmethod
    def sample_noise(self, x):
        pass

    @abstractmethod
    def evaluate_noise_pdf(self, x, data):
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