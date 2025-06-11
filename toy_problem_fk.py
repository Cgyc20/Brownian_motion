from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
from scipy import stats

import particles

class GaussianProb(particles.FeynmanKac):
    def __init__(self, a=0., b=1., T=10):
        self.a, self.b, self.T = a, b, T

    def M0(self, N):
        return stats.norm.rvs(size=N)

    def M(self, t, xp):
        return stats.norm.rvs(loc=xp, size=xp.shape)

    def logG(self, t, xp, x):
        return np.where((x < self.b) & (x > self.a), 0., -np.inf)
    
fk_gp = GaussianProb(a=0., b=1., T=30)
alg = particles.SMC(fk=fk_gp, N=100)
alg.run()

plt.style.use('ggplot')
plt.plot(alg.summaries.logLts)
plt.xlabel('t')
plt.ylabel(r'log-probability')
plt.show()