import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import fftconvolve

# === Physical parameters ===
wavelength = 500  # nm, light wavelength
NA = 1.4          # numerical aperture of objective
refractive_index = 1.33  # water immersion objective (for axial scaling)
n_medium = refractive_index

# Simulation box size in nanometers (1 unit = 1 micron = 1000 nm)
box_size_nm = 1000  # simulation box 1 micron = 1000 nm per axis

# === Simulation parameters ===
number_of_particles = 3
timesteps = 1000
dt = 0.01  # time step, arbitrary units
diffusion_coefficient = 0.001  # arbitrary units

# Spatial grid parameters
grid_size = 100  # pixels per axis
pixel_size_nm = box_size_nm / grid_size  # nm per pixel

# Protein physical size
protein_diameter_nm = 4  # 4 nm diameter protein

# Protein size in pixels:
protein_diameter_pix = 4

# Grid coordinates in nanometers
X_start_nm, X_end_nm = 0, box_size_nm
Y_start_nm, Y_end_nm = 0, box_size_nm
Z_start_nm, Z_end_nm = 0, box_size_nm

x = np.linspace(X_start_nm, X_end_nm, grid_size)
y = np.linspace(Y_start_nm, Y_end_nm, grid_size)
X, Y = np.meshgrid(x, y)

# Focal plane z (axial focus position in nm)
z_focus_nm = box_size_nm / 2  # focal plane at center, 500 nm

# Rayleigh range (depth of focus), simplified
# Rayleigh range z_R = (pi * n * w0^2) / wavelength
# w0 (beam waist) ~ sigma0 = lateral resolution
# Approximate lateral resolution (diffraction limited):
lateral_resolution_nm = 0.21 * wavelength / NA  # sigma0

# Rayleigh range axial depth scale (approximate)
z_R_nm = np.pi * n_medium * lateral_resolution_nm**2 / wavelength

# --- Protein mask generation ---
def circular_protein_mask(size):
    center = size // 2
    Y_, X_ = np.ogrid[:size, :size]
    mask = (X_ - center)**2 + (Y_ - center)**2 <= (center)**2
    return mask.astype(float)

protein_mask = circular_protein_mask(protein_diameter_pix)

# Initialize particle positions (in nm)
positions = np.zeros((number_of_particles, 3, timesteps + 1))
for i in range(number_of_particles):
    positions[i, 0, 0] = np.random.uniform(X_start_nm, X_end_nm)
    positions[i, 1, 0] = np.random.uniform(Y_start_nm, Y_end_nm)
    positions[i, 2, 0] = np.random.uniform(Z_start_nm, Z_end_nm)

# Simulate Brownian motion with reflection at boundaries
for t in range(1, timesteps + 1):
    steps = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(number_of_particles, 3)
    positions[:, :, t] = positions[:, :, t-1] + steps * (box_size_nm * 0.001)  # scaled step size in nm
    # Reflect at boundaries
    for dim in range(3):
        min_bound = [X_start_nm, Y_start_nm, Z_start_nm][dim]
        max_bound = [X_end_nm, Y_end_nm, Z_end_nm][dim]
        positions[:, dim, t] = np.where(
            positions[:, dim, t] < min_bound,
            2 * min_bound - positions[:, dim, t],
            positions[:, dim, t]
        )
        positions[:, dim, t] = np.where(
            positions[:, dim, t] > max_bound,
            2 * max_bound - positions[:, dim, t],
            positions[:, dim, t]
        )

# Gaussian PSF generator
def gaussian_2d(size, sigma):
    center = size // 2
    x_ = np.arange(size) - center
    y_ = np.arange(size) - center
    X_, Y_ = np.meshgrid(x_, y_)
    g = np.exp(-(X_**2 + Y_**2) / (2 * sigma**2))
    return g / g.sum()

# Axial-dependent PSF sigma (in pixels). THis uses the Rayleigh
def sigma_from_z(z_nm, z_focus_nm, sigma0_pix, z_R_nm):
    dz = z_nm - z_focus_nm
    sigma = sigma0_pix * np.sqrt(1 + (dz / z_R_nm)**2)
    return sigma

# Compute sigma0 in pixels from lateral_resolution_nm and pixel_size_nm
sigma0_pix = lateral_resolution_nm / pixel_size_nm

# Setup plot
fig, ax = plt.subplots(figsize=(8, 6))
psf_img = ax.imshow(
    np.zeros((grid_size, grid_size)),
    extent=[X_start_nm, X_end_nm, Y_start_nm, Y_end_nm],
    origin='lower',
    cmap='hot',
    vmin=0,
    vmax=1
)
ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
ax.set_title('Convolved Protein Imaging with Depth-Dependent PSF')

def update(frame):
    img = np.zeros((grid_size, grid_size))
    for i in range(number_of_particles):
        x_nm, y_nm, z_nm = positions[i, :, frame]
        x_pixel = int((x_nm - X_start_nm) / (X_end_nm - X_start_nm) * grid_size)
        y_pixel = int((y_nm - Y_start_nm) / (Y_end_nm - Y_start_nm) * grid_size)

        sigma_pix = sigma_from_z(z_nm, z_focus_nm, sigma0_pix, z_R_nm)
        psf_size = max(7, int(6 * sigma_pix))
        if psf_size % 2 == 0:
            psf_size += 1  # odd size for symmetry

        psf = gaussian_2d(psf_size, sigma_pix)
        convolved = fftconvolve(psf, protein_mask, mode='same')

        half = convolved.shape[0] // 2
        x_start = max(0, x_pixel - half)
        x_end = min(grid_size, x_pixel + half + 1)
        y_start = max(0, y_pixel - half)
        y_end = min(grid_size, y_pixel + half + 1)

        cx_start = half - (x_pixel - x_start)
        cx_end = half + (x_end - x_pixel)
        cy_start = half - (y_pixel - y_start)
        cy_end = half + (y_end - y_pixel)

        # Intensity falloff with axial distance, softer decay
        depth = np.abs(z_nm - z_focus_nm)
        intensity = 1 / (1 + depth / (box_size_nm / 2))  # normalize depth decay

        img[y_start:y_end, x_start:x_end] += intensity * convolved[cy_start:cy_end, cx_start:cx_end]

    if img.max() > 0:
        img /= img.max()

    expected_photons_per_frame = 5
    scaled_img = img * expected_photons_per_frame
    noisy_img = np.random.poisson(scaled_img).astype(float)

    if noisy_img.max() > 0:
        noisy_img /= noisy_img.max()

    psf_img.set_data(noisy_img)
    return [psf_img]

ani = FuncAnimation(fig, update, frames=timesteps + 1, interval=20, blit=True)
plt.colorbar(psf_img, label='Normalized Intensity')
plt.tight_layout()
plt.show()