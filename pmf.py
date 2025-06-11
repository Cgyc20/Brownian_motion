import numpy as np
import matplotlib.pyplot as plt

# --- 1. Constants for the Optical System ---
# These are example values. You can adjust them as needed.
D = 50  # Aperture diameter in mm
u_f = 100  # Focal length of the lens in mm (this is the object distance at which the lens is focused)
s = 2.0  # Scaling factor for the standard deviation in the PSF formula.
         # The variance of the Gaussian PSF will be s * (b/2)^2.

# --- 2. Function to Calculate Circle of Confusion (b) ---
def calculate_circle_of_confusion(D_aperture, focal_length, object_distance_z):
    """
    Calculates the diameter of the circle of confusion (b).

    Args:
        D_aperture (float): Diameter of the lens aperture.
        focal_length (float): Focal length of the lens (object distance for perfect focus).
        object_distance_z (float): Current distance of the object (cell) from the lens.

    Returns:
        float: Diameter of the circle of confusion.
    """
    if focal_length == 0:
        print("Error: Focal length cannot be zero.")
        return 0
    # The formula b = D|(u_f-z)/uf| describes how blur increases as the object
    # moves away from the perfectly focused distance (u_f).
    b = D_aperture * abs(focal_length - object_distance_z) / focal_length
    return b

# --- 3. Function for the 2D Gaussian Point Spread Function (PSF) ---
def gaussian_psf(x, y, b_coc, s_factor):
    """
    Calculates the intensity value of a 2D Gaussian Point Spread Function.

    The given formula is h(x,y) = 1/(2*pi*(b/2)^2 * s) * exp(-(x^2+y^2)/(s*(b/2)^2))
    This means the effective variance (sigma_squared) is s_factor * (b_coc/2)^2.

    Args:
        x (float or np.array): X-coordinate(s) in the image plane.
        y (float or np.array): Y-coordinate(s) in the image plane.
        b_coc (float): Diameter of the circle of confusion.
        s_factor (float): Scaling factor for the variance (from the problem statement).

    Returns:
        float or np.array: Intensity value(s) of the PSF at (x, y).
    """
    # Calculate the effective variance from the circle of confusion and scaling factor
    # This (s_factor * (b_coc/2)^2) is equivalent to sigma_x^2 and sigma_y^2
    # in the standard 2D Gaussian PDF form: 1/(2*pi*sigma_x*sigma_y) * exp(-(x^2/(2*sigma_x^2) + y^2/(2*sigma_y^2)))
    # For a symmetric Gaussian where sigma_x = sigma_y = sigma, this simplifies to
    # 1/(2*pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))
    
    # Based on the user's formula, the denominator inside exp is s*(b/2)^2, so sigma_eff_squared = s*(b/2)^2 / 2
    # The normalization constant is 1/(2*pi * sigma_eff_squared * 2) if we use standard form.
    # Let's derive it directly from the user's formula:
    
    # The user's formula is 1/(2pi (b/2)^2)exp(-(x^2+y^2)/s(b/2)^2)
    # This implies that the variance term in the exponent is s_factor * (b_coc / 2)**2
    # And the pre-factor is 1 / (2 * np.pi * (b_coc / 2)**2)
    # Let's check for consistency. For a 2D Gaussian, if variance is sigma_sq, the form is 1/(2*pi*sigma_sq) * exp(-(x^2+y^2)/(2*sigma_sq))
    # Comparing user's formula:
    # Denominator in exp: s(b/2)^2  <-- This means 2 * sigma_sq_effective = s(b/2)^2, so sigma_sq_effective = s(b/2)^2 / 2
    # Pre-factor: 1/(2pi (b/2)^2) <-- This implies a variance of (b/2)^2 if it were 1/(2*pi*variance).
    # There seems to be a slight inconsistency between the pre-factor and the exponent's variance term if s != 2.

    # Assuming the user's provided formula for h(x,y) is exactly as intended, we will implement it directly:
    effective_variance_in_exponent = s_factor * (b_coc / 2)**2
    
    if effective_variance_in_exponent <= 0:
        # Avoid division by zero or invalid exponent for b_coc = 0
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0

    # Normalization constant for the given formula
    # If the user intends (b/2)^2 to be the variance, then the s_factor in the exponent
    # makes it different. Let's assume the user's literal formula h(x,y) = 1/(2pi (b/2)^2)exp(-(x^2+y^2)/s(b/2)^2)
    # The pre-factor part: 1 / (2 * np.pi * (b_coc / 2)**2)
    # The exponent part: -(x**2 + y**2) / (s_factor * (b_coc / 2)**2)

    # Let's define sigma_effective_for_plot as the actual standard deviation of the Gaussian
    # used for plotting scale, which is sqrt(effective_variance_in_exponent) / sqrt(2)
    # from the standard Gaussian form 1/(2pi*sigma^2) * exp(-(x^2+y^2)/(2*sigma^2))
    # where sigma^2 is the variance.
    # So if (x^2+y^2)/(s*(b/2)^2) is the exponent, then our 2*sigma^2 is s*(b/2)^2
    # sigma_actual_squared = s_factor * (b_coc / 2)**2 / 2

    # Normalization according to the user's exact formula's leading term
    normalization_factor = 1.0 / (2.0 * np.pi * (b_coc / 2.0)**2)
    
    # Calculate the exponent term
    exponent_term = - (x**2 + y**2) / effective_variance_in_exponent
    
    return normalization_factor * np.exp(exponent_term)

# --- 4. Plotting the PSF for Different Distances ---

# Define a range of object distances (z) to simulate defocus
# u_f is the focal length, so z = u_f means perfect focus.
# Distances further from u_f will result in more blur.
z_distances = [
    u_f,       # Perfect focus
    u_f + 10,  # Slightly out of focus (further away)
    u_f + 50,  # More out of focus
    u_f + 150  # Significantly out of focus
]

# Set up the plotting grid
# The grid extent should be dynamic, based on the maximum expected blur (b)
max_b = max(calculate_circle_of_confusion(D, u_f, z) for z in z_distances)
# A good range to visualize a Gaussian is typically +/- 3 to 5 times its standard deviation.
# Here, the standard deviation is sqrt(s) * (b/2).
max_sigma_for_plot = np.sqrt(s) * (max_b / 2)
plot_extent = 5 * max_sigma_for_plot # Extend plot +/- 5 times max_sigma

x_values = np.linspace(-plot_extent, plot_extent, 200)
y_values = np.linspace(-plot_extent, plot_extent, 200)
X, Y = np.meshgrid(x_values, y_values)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

for i, z_dist in enumerate(z_distances):
    ax = axes[i]

    # Calculate circle of confusion for the current z distance
    b_current = calculate_circle_of_confusion(D, u_f, z_dist)

    # Calculate PSF values for the current z distance
    psf_data = gaussian_psf(X, Y, b_current, s)

    # Plot the PSF
    # Using imshow with 'viridis' colormap for intensity visualization
    # extent defines the data limits of the image, so X and Y coordinates match the image
    img = ax.imshow(psf_data, cmap='viridis', origin='lower',
                    extent=[X.min(), X.max(), Y.min(), Y.max()])
    
    ax.set_title(f'PSF at z = {z_dist:.1f} mm\n(b = {b_current:.2f} mm)', fontsize=12)
    ax.set_xlabel('X-position (mm)', fontsize=10)
    ax.set_ylabel('Y-position (mm)', fontsize=10)
    ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
    
    # Add a color bar to each subplot to show intensity scale
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.suptitle('Gaussian Point Spread Function (PSF) at Varying Distances (Defocus)',
             y=1.02, fontsize=16) # Add a super title for the entire figure
plt.show()

# --- Example of getting a single PSF value ---
# Let's say you want the PSF value at (x=0, y=0) for z=u_f+10
z_example = u_f + 10
b_example = calculate_circle_of_confusion(D, u_f, z_example)
psf_at_center = gaussian_psf(0, 0, b_example, s)
print(f"\nExample: PSF intensity at center (0,0) for z = {z_example:.1f} mm: {psf_at_center:.4e}")
print(f"Corresponding Circle of Confusion (b) = {b_example:.2f} mm")


from scipy.signal import convolve2d

def create_circle_particle(size=9, radius=3):
    img = np.zeros((size, size))
    cx, cy = size // 2, size // 2
    for x in range(size):
        for y in range(size):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                img[y, x] = 1.0
    return img

def gaussian_psf(size=9, sigma=1.5):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

particle = create_circle_particle(size=9, radius=3)
psf = gaussian_psf(size=9, sigma=2.0)

blurred = convolve2d(particle, psf, mode='same', boundary='fill', fillvalue=0)

plt.subplot(1,3,1)
plt.title("Particle")
plt.imshow(particle, cmap='gray')
plt.subplot(1,3,2)
plt.title("PSF")
plt.imshow(psf, cmap='gray')
plt.subplot(1,3,3)
plt.title("Blurred")
plt.imshow(blurred, cmap='gray')
plt.tight_layout()
plt.show()