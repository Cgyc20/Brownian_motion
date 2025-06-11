import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
number_of_particles = 20
timesteps = 1000
dt = 0.01
diffusion_coefficient = 0.001

# Boundaries
X_start, X_end = 0, 1
Y_start, Y_end = 0, 1
Z_start, Z_end = 0, 1

# PSF parameters
D = 0.08   # Aperture diameter
uf = 0.4      # Focal plane at z=0.5 (middle of Z_start, Z_end)

# Initialize particle positions
positions = np.zeros((number_of_particles, 3, timesteps + 1))
for i in range(number_of_particles):
    positions[i, 0, 0] = np.random.uniform(X_start, X_end)  # x
    positions[i, 1, 0] = np.random.uniform(Y_start, Y_end)  # y
    positions[i, 2, 0] = np.random.uniform(Z_start, Z_end)  # z

# Simulate Brownian motion with reflection at boundaries
for t in range(1, timesteps + 1):
    steps = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(number_of_particles, 3)
    positions[:, :, t] = positions[:, :, t-1] + steps
    # Reflect particles at boundaries
    for dim in range(3):
        positions[:, dim, t] = np.where(
            positions[:, dim, t] < [X_start, Y_start, Z_start][dim],
            2 * [X_start, Y_start, Z_start][dim] - positions[:, dim, t],
            positions[:, dim, t]
        )
        positions[:, dim, t] = np.where(
            positions[:, dim, t] > [X_end, Y_end, Z_end][dim],
            2 * [X_end, Y_end, Z_end][dim] - positions[:, dim, t],
            positions[:, dim, t]
        )

# Prepare grid for rendering
grid_size = 100
x = np.linspace(X_start, X_end, grid_size)
y = np.linspace(Y_start, Y_end, grid_size)
X, Y = np.meshgrid(x, y)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 6))
psf_img = ax.imshow(
    np.zeros((grid_size, grid_size)),
    extent=[X_start, X_end, Y_start, Y_end],
    origin='lower',
    cmap='hot',
    vmin=0,
    vmax=1
)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Brownian Motion with Depth-Dependent Blur and Brightness')

def gaussian_2d(x, y, x0, y0, sigma):
    """2D Gaussian PSF centered at (x0, y0) with width sigma."""
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def sigma_from_z(z, D, uf):
    """Compute PSF width (sigma) based on z-position."""
    return 0.5 * D * np.abs((uf - z) / uf)

def update(frame):
    img = np.zeros((grid_size, grid_size))
    eps = 1e-6  # Small value to avoid division by zero
    for i in range(number_of_particles):
        z = positions[i, 2, frame]
        sigma = sigma_from_z(z, D, uf)
        # Intensity scales as 1/(distance_from_focus² + eps)
        intensity = 0.5 / ((z - uf)**2 + 0.02)
        img += intensity * gaussian_2d(X, Y, positions[i, 0, frame], positions[i, 1, frame], sigma)
    # Normalize to [0, 1]
    if img.max() > 0:
        img /= img.max()
    psf_img.set_data(img)
    return [psf_img]

# Animate
ani = FuncAnimation(fig, update, frames=timesteps + 1, interval=20, blit=True)
plt.colorbar(psf_img, label='Normalized Intensity')
plt.tight_layout()
plt.show()