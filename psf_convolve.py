import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import fftconvolve

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
uf = 0.4   # Focal plane at z=0.5 (middle of Z_start, Z_end)

# Grid settings
grid_size = 100
x = np.linspace(X_start, X_end, grid_size)
y = np.linspace(Y_start, Y_end, grid_size)
X, Y = np.meshgrid(x, y)
pixel_size = (X_end - X_start) / grid_size

# Create circular protein mask
def circular_protein_mask(size):
    """Create a circular binary protein mask of a given size (e.g., 4x4 pixels)."""
    center = size // 2
    Y, X = np.ogrid[:size, :size]
    mask = (X - center)**2 + (Y - center)**2 <= (center)**2
    return mask.astype(float)

protein_mask = circular_protein_mask(4)

# Initialize particle positions
positions = np.zeros((number_of_particles, 3, timesteps + 1))
for i in range(number_of_particles):
    positions[i, 0, 0] = np.random.uniform(X_start, X_end)
    positions[i, 1, 0] = np.random.uniform(Y_start, Y_end)
    positions[i, 2, 0] = np.random.uniform(Z_start, Z_end)

# Simulate Brownian motion with reflection at boundaries
for t in range(1, timesteps + 1):
    steps = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(number_of_particles, 3)
    positions[:, :, t] = positions[:, :, t-1] + steps
    # Reflect at boundaries
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

# Set up plot
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
ax.set_title('Convolved Protein Imaging with Depth-Dependent PSF')

# Gaussian PSF generator
def gaussian_2d(size, sigma):
    """Return a square 2D Gaussian of given size and sigma."""
    center = size // 2
    x = np.arange(size) - center
    y = np.arange(size) - center
    X, Y = np.meshgrid(x, y)
    g = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return g / g.sum()

# Sigma as function of z-depth
def sigma_from_z(z, D, uf):
    """Compute PSF width (sigma in pixels) based on z-position."""
    sigma_world = 1 * D * np.abs((uf - z) / uf)
    return sigma_world / pixel_size  # convert to pixels

def update(frame):
    img = np.zeros((grid_size, grid_size))
    for i in range(number_of_particles):
        x_world, y_world, z = positions[i, :, frame]
        x_pixel = int((x_world - X_start) / (X_end - X_start) * grid_size)
        y_pixel = int((y_world - Y_start) / (Y_end - Y_start) * grid_size)
        
        sigma_pix = sigma_from_z(z, D, uf)
        psf_size = max(7, int(6 * sigma_pix))  # ensure coverage
        if psf_size % 2 == 0: psf_size += 1  # make odd for symmetry

        # Create convolved object: PSF * protein
        psf = gaussian_2d(psf_size, sigma_pix)
        convolved = fftconvolve(psf, protein_mask, mode='same')

        # Add to main image at (x_pixel, y_pixel)
        half = convolved.shape[0] // 2
        x_start = max(0, x_pixel - half)
        x_end = min(grid_size, x_pixel + half + 1)
        y_start = max(0, y_pixel - half)
        y_end = min(grid_size, y_pixel + half + 1)

        cx_start = half - (x_pixel - x_start)
        cx_end = half + (x_end - x_pixel)
        cy_start = half - (y_pixel - y_start)
        cy_end = half + (y_end - y_pixel)

        depth = np.abs(z - uf)
        intensity = 1 / (1 + 1 * depth)  # shallower falloff
        img[y_start:y_end, x_start:x_end] += intensity * convolved[cy_start:cy_end, cx_start:cx_end]

    if img.max() > 0:
        img /= img.max()
    psf_img.set_data(img)
    return [psf_img]

# Animate
ani = FuncAnimation(fig, update, frames=timesteps + 1, interval=20, blit=True)
plt.colorbar(psf_img, label='Normalized Intensity')
plt.tight_layout()
plt.show()