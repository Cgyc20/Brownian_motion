import numpy as np
import particles
import matplotlib.pyplot as plt
from particles.collectors import Moments
from molecule_model_2d import MoleculeBootstrapPF2D

np.random.seed(3)

dt=0.1
diffusion_coef = 0.1
init_mean = np.array([0.5, 0.5])
init_cov = 0.01 * np.eye(2)
box = np.array([[0.0, 1.], [0.01, 1.01]])
pixel_width = 0.01
T = 30
multinom_samples = 1000

state_space_model = MoleculeBootstrapPF2D(init_mean=init_mean,
                                     init_cov=init_cov,
                                     diffusion_coef=diffusion_coef,
                                     dt=dt,
                                     box=box,
                                     T=T,
                                     pixel_width=pixel_width,
                                     multinom_samples=multinom_samples)

true_path = state_space_model.gen_path()

# Plot true path
fig, ax = plt.subplots()
ax.plot(true_path[:, 0], true_path[:, 1], marker='o', markersize=3, linewidth=1)
ax.set_xlim(box[0])
ax.set_ylim(box[1])
ax.set_aspect('equal')
ax.set_title("True molecule trajectory")
plt.grid(True)
plt.show()

observed_data = state_space_model.generate_observations(path=true_path)

t = 5
pixel_centers = 0.5 * (np.arange(box[0, 0], box[0, 1] + pixel_width, pixel_width)[:-1] +
                       np.arange(box[0, 0], box[0, 1] + pixel_width, pixel_width)[1:])
plt.bar(pixel_centers, observed_data[t], width=pixel_width)
plt.title(f'Observation at time t={t}')
plt.xlabel("Position")
plt.ylabel("Photon counts")
plt.show()

state_space_model.data = observed_data

alg = particles.SMC(fk=state_space_model, N=1000,
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