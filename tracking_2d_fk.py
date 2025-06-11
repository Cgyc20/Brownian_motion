import numpy as np
from particles import FeynmanKac
import particles
from scipy.stats import multivariate_normal
import scipy.special
import matplotlib.pyplot as plt
from particles.collectors import Moments

np.random.seed(3)


class ReflectedBootstrapMolecule(FeynmanKac):
    def __init__(self, init_mean, init_cov, diffusion_coef, dt, box, T, data, pixel_width):
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.diffusion_coef = diffusion_coef
        self.dt = dt
        self.box = box  # shape (2, 2): [[xmin, xmax], [ymin, ymax]]
        self.T = T
        self.data = data
        self.pixel_width = pixel_width

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

    def logG(self, t, xp, x):
        """
        Compute log-likelihoods of particles under multinomial observation model.

        Parameters:
        - t: current time index
        - xp: previous particle states (unused in bootstrap filter)
        - x: current particle states, shape (N, 2) -> each row: [mean, scale]
        - data: full observation array of shape (T, num_pixels)

        Returns:
        - log_likelihoods: array of shape (N,) with log-likelihood for each particle
        """
        observed = self.data[t]
        total_counts = np.sum(observed)
        pixel_edges = np.arange(self.box[0, 0], self.box[0, 1] + self.pixel_width, self.pixel_width)
        pixel_centers = 0.5 * (pixel_edges[:-1] + pixel_edges[1:])
        num_pixels = len(pixel_centers)

        N = x.shape[0]
        log_likelihoods = np.zeros(N)

        for i in range(N):
            mu, scale = x[i]
            denom = (scale)**2 + 1e-12

            # Gaussian profile over pixels
            gauss = (1. / (np.sqrt(2 * np.pi * denom)) *
                    np.exp(-0.5 * ((pixel_centers - mu) ** 2) / denom))
            probs = gauss / np.sum(gauss)

            # Avoid log(0) issues
            probs = np.clip(probs, 1e-12, 1.0)
            probs = probs / np.sum(probs)  # Renormalize after clipping

            # Multinomial log-PMF
            log_coef = scipy.special.gammaln(total_counts + 1) - np.sum(scipy.special.gammaln(observed + 1))
            log_prob = np.sum(observed * np.log(probs))
            log_likelihoods[i] = log_coef + log_prob

        return log_likelihoods
    
dt=0.1
diffusion_coef = 0.1
init_mean = np.array([0.5, 0.5])
init_cov = 0.01 * np.eye(2)
box = np.array([[0.0, 1.], [0.01, 1.01]])
pixel_width = 0.01
T = 30

def gen_path(T, init_mean, init_cov, diffusion_coef, dt, box):
    def reflect(x):
        x_reflected = np.copy(x)
        for i in range(x.shape[1]):
            lower, upper = box[i]
            x_reflected[:, i] = np.where(x_reflected[:, i] < lower,
                                         2 * lower - x_reflected[:, i],
                                         x_reflected[:, i])
            x_reflected[:, i] = np.where(x_reflected[:, i] > upper,
                                         2 * upper - x_reflected[:, i],
                                         x_reflected[:, i])
        return x_reflected

    path = []
    for t in range(T):
        if t == 0:
            sample = np.random.multivariate_normal(init_mean, init_cov, size=1)
        else:
            noise = np.random.normal(loc=0.0,
                                     scale=np.sqrt(2 * diffusion_coef * dt),
                                     size=(1, 2))  # 2D particle
            sample = path[-1][np.newaxis, :] + noise
        sample = reflect(sample)[0]  # Reflect and extract 1D
        path.append(sample)
    return np.array(path)

true_path = gen_path(T, init_mean, init_cov, diffusion_coef, dt, box)

# Plot the trajectory
fig, ax = plt.subplots()
ax.plot(true_path[:, 0], true_path[:, 1], marker='o', markersize=3, linewidth=1)
ax.set_xlim(box[0])
ax.set_ylim(box[1])
ax.set_aspect('equal')
ax.set_title("True molecule trajectory")
plt.grid(True)
plt.show()

def generate_observations(true_path, box, pixel_width, const=1e-3):
    """
    Generate synthetic observations from a reflected Brownian motion path.

    Parameters:
    - true_path: array of shape (T, 2), each row is [mean, scale]
    - box: array of shape (2, 2), e.g., [[xmin, xmax], [ymin, ymax]]
    - pixel_width: float, width of each pixel (defines resolution of image)
    - const: float, small value to regularize scale denominator

    Returns:
    - observations: list of 1D numpy arrays (length determined by box and pixel width)
    """
    T = true_path.shape[0]
    pixel_edges = np.arange(box[0, 0], box[0, 1] + pixel_width, pixel_width)
    pixel_centers = 0.5 * (pixel_edges[:-1] + pixel_edges[1:])
    num_pixels = len(pixel_centers)

    observations = []

    for t in range(T):
        mu, scale = true_path[t]
        denom = (scale)**2

        # Construct Gaussian profile
        gauss = (1. / (np.sqrt(2 * np.pi * denom)) *
                 np.exp(-0.5 * ((pixel_centers - mu) ** 2) / denom))
        gauss = gauss / sum(gauss[:-1])

        # Normalize intensity to a total count (optional: scale up for visibility)
        #intensity = gauss * num_pixels / np.sum(gauss) * 20.  # 20 photons on average

        # Sample from Poisson distribution
        #y_t = np.random.poisson(lam=gauss)
        y_t = np.random.multinomial(100000, gauss)
        #y_t = y_t/sum(y_t[:-1])
        observations.append(y_t)

    return observations


# Generate data
observations = generate_observations(true_path=true_path,
                                     box=box,
                                     pixel_width=pixel_width)

t = 5
pixel_centers = 0.5 * (np.arange(box[0, 0], box[0, 1] + pixel_width, pixel_width)[:-1] +
                       np.arange(box[0, 0], box[0, 1] + pixel_width, pixel_width)[1:])
plt.bar(pixel_centers, observations[t], width=pixel_width)
plt.title(f'Observation at time t={t}')
plt.xlabel("Position")
plt.ylabel("Photon counts")
plt.show()

fk_model = ReflectedBootstrapMolecule(
    init_mean=init_mean,
    init_cov=init_cov,
    diffusion_coef=diffusion_coef,
    dt=dt,
    box=box,
    T=T,
    data=observations,
    pixel_width=pixel_width
)

alg = particles.SMC(fk=fk_model, N=1000,
                   collect=[Moments()], store_history=True)
alg.run()

fig, ax = plt.subplots()

# Plot true trajectory
ax.plot(true_path[:, 0], true_path[:, 1], marker='o', markersize=3,
        linewidth=1, label='True trajectory', color='blue')

# Extract inferred means from particle filter
inferred_path = np.array([m['mean'] for m in alg.summaries.moments])  # shape: (T, 2)

# Plot inferred trajectory
ax.plot(inferred_path[:, 0], inferred_path[:, 1], marker='x', markersize=3,
        linewidth=1, label='Inferred trajectory', color='red')

# Formatting
ax.set_xlim(box[0])
ax.set_ylim(box[1])
ax.set_aspect('equal')
ax.set_title("True vs Inferred Molecule Trajectory")
ax.legend()
plt.grid(True)
plt.show()
