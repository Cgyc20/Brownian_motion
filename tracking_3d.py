import numpy as np
import particles
import matplotlib.pyplot as plt
from particles.collectors import Moments
from molecule_model_3d import MoleculeBootstrapPF3D

np.random.seed(3)

dt=0.1
diffusion_coef = 0.1
init_mean = np.array([0.5, 0.5, 0.5])
init_cov = 0.01 * np.eye(3)
box = np.array([[0.0, 1.], [0.0, 1.], [0.01, 1.01]])
pixel_width = 0.01
T = 30
multinom_samples = 100000

state_space_model = MoleculeBootstrapPF3D(init_mean=init_mean,
                                         init_cov=init_cov,
                                         diffusion_coef=diffusion_coef,
                                         dt=dt,
                                         box=box,
                                         T=T,
                                         pixel_width=pixel_width,
                                         multinom_samples=multinom_samples)

true_path = state_space_model.gen_path()
# Plot true path in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2], marker='o', markersize=3, linewidth=1)
ax.set_xlim(box[0])
ax.set_ylim(box[1])
ax.set_zlim(box[2])
ax.set_title("True molecule trajectory")
plt.grid(True)
plt.show()

observed_data = state_space_model.generate_observations(path=true_path)
t=5
#Plot observed data at time t
obs_time = observed_data[t]
plt.figure(figsize=(6, 6))
plt.imshow(obs_time, cmap='gray', origin='lower')  # or 'hot', 'viridis' for color
plt.title(f'Observed Data at t = {t}')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.colorbar(label='Intensity')
plt.show()

state_space_model.data = observed_data

alg = particles.SMC(fk=state_space_model, N=1000,
                   collect=[Moments()], store_history=True)
alg.run()

#Plot paths in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2], marker='o', markersize=3, linewidth=1, label='True trajectory', color='blue')

#Extract the inferred means from the algorithm
inferred_path = np.array([m['mean'] for m in alg.summaries.moments])  # shape: (T, 3)

ax.plot(inferred_path[:, 0], inferred_path[:, 1], inferred_path[:, 2], marker='x', markersize=3, linewidth=1, label='Inferred trajectory', color='red')

#Formatting
ax.set_xlim(box[0])
ax.set_ylim(box[1])
ax.set_zlim(box[2])
ax.set_title("True vs Inferred Molecule Trajectory")
ax.legend()
plt.grid(True)
plt.show()