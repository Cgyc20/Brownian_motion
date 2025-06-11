import numpy as np
import matplotlib.pyplot as plt

# PSF and plotting setup
D = 0.08
uf = 0.4
grid_size = 100
X_start, X_end = 0, 1
Y_start, Y_end = 0, 1
x0, y0 = 0.5, 0.5  # Same particle position

z_values = [0.5, 0.9]  # In-focus and out-of-focus depths
x_line = np.linspace(0, 1, grid_size)
x, y = np.linspace(X_start, X_end, grid_size), np.linspace(Y_start, Y_end, grid_size)
X, Y = np.meshgrid(x, y)

def gaussian_2d(x, y, x0, y0, sigma):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def sigma_from_z(z, D, uf):
    return 0.5 * D * np.abs((uf - z) / uf)

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

for i, z in enumerate(z_values):
    sigma = sigma_from_z(z, D, uf)
    intensity = 0.5 / ((z - uf)**2 + 0.02)

    # 1D Gaussian slice (centered at x0)
    psf_1d = np.exp(-((x_line - x0)**2) / (2 * sigma**2))
    axes[0, i].plot(x_line, psf_1d)
    axes[0, i].set_title(f'1D PSF Slice at z={z:.2f}')
    axes[0, i].set_xlabel('x')
    axes[0, i].set_ylabel('Intensity')

    # 2D PSF image
    psf_2d = intensity * gaussian_2d(X, Y, x0, y0, sigma)
    im = axes[1, i].imshow(psf_2d, extent=[0.3, 0.7, 0.3, 0.7], origin='lower', cmap='hot')
    axes[1, i].set_title(f'2D PSF at z={z:.2f}')
    fig.colorbar(im, ax=axes[1, i], fraction=0.046)

plt.tight_layout()
plt.show()