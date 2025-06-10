import numpy as np
import matplotlib.pyplot as plt

number_of_particles = 10
timesteps = 1000
dt = 0.01  # time step size
diffusion_coefficient = 1.0

# Initialize positions: shape (number_of_particles, 2)
positions = np.zeros((number_of_particles, 2, timesteps + 1))

# Set initial positions (e.g., all at center)
positions[:, 0, 0] = 0.5  # x
positions[:, 1, 0] = 0.5  # y

for t in range(1, timesteps + 1):
    # Random steps: sqrt(2Ddt) * N(0,1)
    steps = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(number_of_particles, 2)
    positions[:, :, t] = positions[:, :, t-1] + steps

# Plot trajectories
for i in range(number_of_particles):
    plt.plot(positions[i, 0, :], positions[i, 1, :])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Brownian Motion of N Particles')
plt.show()