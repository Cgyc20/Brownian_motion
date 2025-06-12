import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import fftconvolve
import matplotlib.colors as mcolors
# === Physical parameters ===
wavelength = 100  # nm, light wavelength
NA = 1.4          # numerical aperture of objective
refractive_index = 1.33  # water immersion objective (for axial scaling)
n_medium = refractive_index

# Simulation box size in nanometers (1 unit = 1 micron = 1000 nm)
box_size_nm = 1000  # simulation box 1 micron = 1000 nm per axis

X_start_nm, X_end_nm = 0, box_size_nm
Y_start_nm, Y_end_nm = 0, box_size_nm
Z_start_nm, Z_end_nm = 0, box_size_nm

# === Simulation parameters ===
number_of_particles = 10
timesteps = 1000
dt = 0.1  # time step, arbitrary units
diffusion_coefficient = 100 # arbitrary units

# Spatial grid parameters
grid_size = 100  # pixels per axis
pixel_size_nm = box_size_nm / grid_size  # nm per pixel



switch_rates = [0.05 , 0.05, 0.025, 0.025, 0.07] #F -> Q, Q -> F, F -> H, Q -> H, H -> Z
relative_intensity_rate = [1, 0.2, 0.5, 0] #Relative intensity of F, Q, H, Z states
# Protein physical size
protein_diameter_nm = 4  # 4 nm diameter protein
protein_diameter_pix = 1  # in pixels (used for mask size)

# Grid coordinates in nanometers
x = np.linspace(0, box_size_nm, grid_size)
y = np.linspace(0, box_size_nm, grid_size)
states = np.zeros((number_of_particles, timesteps+1),dtype = int) #Define at zero

X, Y = np.meshgrid(x, y)

# Focal plane z (axial focus position in nm)
z_focus_nm = box_size_nm / 2  # focal plane at center, 500 nm

# Approximate lateral resolution (diffraction limited)
lateral_resolution_nm = 0.21 * wavelength / NA  # sigma0 in nm

# Compute sigma0 in pixels
sigma0_pix = lateral_resolution_nm / pixel_size_nm

sigma0_pix /= 2 #Make the initial sigma slightly smaller
# Rayleigh range z_R = (pi * n * sigma0^2) / wavelength
z_R_nm = np.pi * n_medium * lateral_resolution_nm**2 / wavelength

z_R_nm *= 4 #Make it a bit larger! 
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
    positions[i, 0, 0] = np.random.uniform(0, box_size_nm)
    positions[i, 1, 0] = np.random.uniform(0, box_size_nm)
    positions[i, 2, 0] = np.random.uniform(0, box_size_nm)


# Overwrite z-coordinate to keep the first particle at z = 500 nm (focal plane)
positions[0, 2, :] = 500

# Simulate Brownian motion with reflection at boundaries
for t in range(1, timesteps + 1):
    steps = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(number_of_particles, 3)
    positions[:, :, t] = positions[:, :, t-1] + steps   # scaled step size in nm
    # Reflect at boundaries
    positions[:, :, t] = positions[:, :, t-1] + steps
    positions[:, 0, t] = np.where(positions[:, 0, t] < X_start_nm, np.abs(positions[:, 0, t]), positions[:, 0, t])
    positions[:, 1, t] = np.where(positions[:, 1, t] < Y_start_nm, np.abs(positions[:, 1, t]), positions[:, 1, t])
    positions[:, 0, t] = np.where(positions[:, 0, t] > X_end_nm, X_end_nm - (positions[:, 0, t] - X_end_nm), positions[:, 0, t])
    positions[:, 1, t] = np.where(positions[:, 1, t] > Y_end_nm, Y_end_nm - (positions[:, 1, t] - Y_end_nm), positions[:, 1, t])

#Ranging over the states (Quenched, unquenched, First bleaching, second bleaching)
def switching(num_particles, init_state, switch_rates, timesteps, dt):
    """
    Simulates the switching of particles between four states: full, quenched, half, and 
    zero over a number of timesteps."""
    # Parameters:
    # num_particles: Number of particles in the system
    # init_state: Initial state of the particles (array of size num_particles)
    # 0: F, 1 Q, 2: H, 3: Z
    # switch_rates: Array of rates for switching between states [full_rate, quenched_rate, half_rate, zero_rate]
    # timesteps: Number of timesteps to simulate
    #dt: time step size

    #Get information from the input parameters
    init = np.copy(init_state)
    full_rate = switch_rates[0]*dt #F -> Q
    quenched_rate = switch_rates[1]*dt #Q -> F
    half_rateF = switch_rates[2]*dt #F -> H
    half_rateQ = switch_rates[3]*dt #Q -> H
    zero_rate = switch_rates[4]*dt #H -> Z

    # Initialise arrays to store the state of particles at each timestep
    states = np.zeros((num_particles, timesteps+1))
    states[:, 0] = init

    #iterate over timesteps to get the state of each particle
    for i in range(timesteps):
        for j in range(num_particles):
            #Full state
            if states[j, i] == 0:
                #generate random number
                r = np.random.rand()
                #Switch to quenched state
                if r < full_rate:
                    states[j, i+1] = 1
                #Switch to half state
                elif r < full_rate + half_rateF:
                    states[j, i+1] = 2
                else:
                    states[j, i+1] = 0
            
            #Quenched state
            if states[j, i] == 1:
                #generate random number
                r = np.random.rand()
                #Switch to full state
                if r < quenched_rate:
                    states[j, i+1] = 0
                #Switch to half state
                elif r < quenched_rate + half_rateQ:
                    states[j, i+1] = 2
                else:
                    states[j, i+1] = 1
            
            #Half state
            if states[j, i] == 2:
                #generate random number
                r = np.random.rand()
                #Switch to zero state
                if r < zero_rate:
                    states[j, i+1] = 3
                else:
                    states[j, i+1] = 2
            #Can't leave zero state
            if states[j, i] == 3:
                states[j, i+1] = 3
            
    return states

init_state = np.zeros(number_of_particles,dtype = int)
states = switching(num_particles=number_of_particles, init_state=init_state, switch_rates=switch_rates, timesteps=timesteps, dt = dt)

print(states)
# Gaussian PSF generator
def gaussian_2d(size, sigma):
    ##The size is just how much we are plotting! 3 standard deviations from the centre are 99% of all data
    center = size // 2
    x_ = np.arange(size) - center
    y_ = np.arange(size) - center
    X_, Y_ = np.meshgrid(x_, y_)
    g = np.exp(-(X_**2 + Y_**2) / (2 * sigma**2))
    return g / g.sum()

# Enhanced sigma calculation including particle size contribution
def sigma_from_z(z_nm, z_focus_nm, sigma0_pix, z_R_nm, d_nm, pixel_size_nm):
    dz = z_nm - z_focus_nm
    sigma_psf = sigma0_pix * np.sqrt(1 + (dz / z_R_nm) ** 2)
    sigma_par = (d_nm / (2 * np.sqrt(2 * np.log(2)))) / pixel_size_nm
    return sigma_psf + sigma_par

# --- Precompute global max intensity for normalization ---
print("Precomputing global maximum intensity for normalization...")
global_max = 0
for frame in range(timesteps + 1):
    img = np.zeros((grid_size, grid_size))
    for i in range(number_of_particles):
        x_nm, y_nm, z_nm = positions[i, :, frame]
        x_pixel = int((x_nm / box_size_nm) * grid_size)
        y_pixel = int((y_nm / box_size_nm) * grid_size)
        sigma_pix = sigma_from_z(z_nm, z_focus_nm, sigma0_pix, z_R_nm, protein_diameter_nm, pixel_size_nm) #Work out height dependent sigma
        psf_size = int(6 * sigma_pix) #6 as this is 3 SD's from the mean either side. 
        if psf_size % 2 == 0:
            psf_size += 1 
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
        depth = abs(z_nm - z_focus_nm)
        intensity = 1 / (1 + (depth / (box_size_nm*2)))
        img[y_start:y_end, x_start:x_end] += intensity * convolved[cy_start:cy_end, cx_start:cx_end]
    expected_photons_per_frame = 5000
    scaled_img = img * expected_photons_per_frame
    noisy_img = np.random.poisson(scaled_img).astype(float)
    frame_max = noisy_img.max()
    if frame_max > global_max:
        global_max = frame_max
print(f"Global maximum intensity: {global_max}")

# Setup plot
norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=1)
fig, ax = plt.subplots(figsize=(8, 6))
psf_img = ax.imshow(
    np.zeros((grid_size, grid_size)),
    extent=[0, box_size_nm, 0, box_size_nm],
    origin='lower',
    cmap='hot',
    norm=norm
)
ax.set_xlabel('X (nm)')
ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
title_text = ax.set_title('Convolved Protein Imaging with Depth-Dependent PSF')

def update(frame):
    img = np.zeros((grid_size, grid_size))
    for i in range(number_of_particles):

        
        
        relative_intensity_current = relative_intensity_rate[int(states[i, frame])]


        print(relative_intensity_current)
        x_nm, y_nm, z_nm = positions[i, :, frame]
        x_pixel = int((x_nm / box_size_nm) * grid_size)
        y_pixel = int((y_nm / box_size_nm) * grid_size)
        sigma_pix = sigma_from_z(z_nm, z_focus_nm, sigma0_pix, z_R_nm, protein_diameter_nm, pixel_size_nm)
        psf_size =  int(6 * sigma_pix)
        if psf_size % 2 == 0:
            psf_size += 1
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
        depth = abs(z_nm - z_focus_nm)
        intensity = relative_intensity_current/ (1 + (depth / (box_size_nm / 2)))
        
        img[y_start:y_end, x_start:x_end] += intensity * convolved[cy_start:cy_end, cx_start:cx_end]
    expected_photons_per_frame = 10000
    scaled_img = img * expected_photons_per_frame
    noisy_img = np.random.poisson(scaled_img).astype(float)
    # --- Global normalization ---
    if global_max > 0:
        noisy_img /= global_max
    current_time = frame * dt
    title_text.set_text(f"Convolved Protein Imaging\nTime = {current_time:.2f} units")
    psf_img.set_data(noisy_img)
    return [psf_img, title_text]

ani = FuncAnimation(fig, update, frames=timesteps + 1, interval=20, blit=False)
plt.colorbar(psf_img, label='Normalized Intensity')
plt.tight_layout()
plt.show()