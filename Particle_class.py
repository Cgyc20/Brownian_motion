import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import fftconvolve
import matplotlib.colors as mcolors
from typing import Tuple, Optional, List
from dataclasses import dataclass
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

@dataclass
class SimulationParameters:
    """
    Configuration parameters for the fluorescence microscopy simulation.
    
    Attributes:
        wavelength: Light wavelength in nanometers
        numerical_aperture: Numerical aperture of the microscope objective
        refractive_index: Refractive index of the immersion medium
        box_size_nm: Size of the simulation box in nanometers
        number_of_particles: Number of fluorescent particles to simulate
        timesteps: Number of simulation timesteps
        dt: Time step size in arbitrary units
        diffusion_coefficient: Brownian motion diffusion coefficient
        grid_size: Number of pixels per axis in the simulation grid
        switch_rates: State transition rates [F→Q, Q→F, F→H, Q→H, H→Z]
        relative_intensities: Relative fluorescence intensities [F, Q, H, Z states]
        protein_diameter_nm: Physical diameter of protein in nanometers
        expected_photons: Expected photons per frame for noise simulation
    """
    # Optical parameters
    wavelength: float = 100.0  # nm
    numerical_aperture: float = 1.4
    refractive_index: float = 1.33  # water immersion
    
    # Simulation box
    box_size_nm: float = 1000.0  # 1 micron
    
    # Particle dynamics
    number_of_particles: int = 10
    total_time : float = 100.0  # Total simulation time in arbitrary units
    
    dt: float = 0.1
    timesteps: int = int(np.ceil(total_time/ dt) ) # Number of timesteps derived from total_time and dt
    diffusion_coefficient: float = 100.0
    
    # Imaging parameters
    grid_size: int = 100
    switch_rates: List[float] = None  # Will be set in __post_init__
    relative_intensities: List[float] = None  # Will be set in __post_init__
    protein_diameter_nm: float = 4.0
    expected_photons: int = 10000
    

    def __post_init__(self):
        if self.switch_rates is None:
            # Default rates: [F→Q, Q→F, F→H, Q→H, H→Z]
            self.switch_rates = [0.05, 0.05, 0.025, 0.025, 0.07]
        if self.relative_intensities is None:
            # Default intensities: [F, Q, H, Z states]
            self.relative_intensities = [2.0, 0.2, 0.5, 0.0]


class FluorescenceMicroscopySimulator:
    """
    A comprehensive simulator for fluorescence microscopy with Brownian motion,
    photoswitching dynamics, and realistic point spread function modeling.
    """

    def __init__(self, params: Optional[SimulationParameters] = None):
        self.params = params if params is not None else SimulationParameters()
        (
            self.lateral_resolution_nm,
            self.pixel_size_nm,
            self.sigma0_pix,
            self.z_R_nm,
            self.z_focus_nm
        ) = self.calculate_optical_parameters(self.params)
        self.X_grid, self.Y_grid = self.initialize_spatial_grid(self.params)
        self.protein_mask = self.create_protein_mask(self.params)
        self.positions, self.states = self.initialize_particles(self.params)
        self.positions = self.simulate_trajectories(self.positions, self.params)
        self.states = self.simulate_photoswitching(self.states, self.params)
        self.psf_norm = self.precompute_normalization(self.sigma0_pix)

    @staticmethod
    def calculate_optical_parameters(params: SimulationParameters):
        lateral_resolution_nm = 0.21 * params.wavelength / params.numerical_aperture
        pixel_size_nm = params.box_size_nm / params.grid_size
        sigma0_pix = (lateral_resolution_nm / pixel_size_nm) / 2
        z_R_nm = (np.pi * params.refractive_index * lateral_resolution_nm**2 / params.wavelength) * 4
        z_focus_nm = params.box_size_nm / 2
        return lateral_resolution_nm, pixel_size_nm, sigma0_pix, z_R_nm, z_focus_nm

    @staticmethod
    def initialize_spatial_grid(params: SimulationParameters):
        X = np.linspace(0, params.box_size_nm, params.grid_size)
        Y = np.linspace(0, params.box_size_nm, params.grid_size)
        return np.meshgrid(X, Y)

    @staticmethod
    def create_protein_mask(params: SimulationParameters):
        centre = params.protein_diameter_nm // 2
        Y_, X_ = np.ogrid[:params.protein_diameter_nm, :params.protein_diameter_nm]
        mask = (X_ - centre)**2 + (Y_ - centre)**2 <= (centre)**2
        return mask.astype(float)


    def initialize_particles(self, params: SimulationParameters):
        positions = np.zeros((params.number_of_particles, 3, params.timesteps + 1))
        for i in range(params.number_of_particles):
            positions[i, 0, 0] = np.random.uniform(0, params.box_size_nm)
            positions[i, 1, 0] = np.random.uniform(0, params.box_size_nm)
            positions[i, 2, 0] = np.random.uniform(0, params.box_size_nm)
        
        # positions[:,2,0] = 500
        #States are Full, quenches, half bleached and full bleached. 
        states = np.zeros((params.number_of_particles, params.timesteps + 1), dtype=int)
        states[:, 0] = 0
        return positions, states

    @staticmethod
    def simulate_trajectories(positions: np.ndarray, params: SimulationParameters):
        for t in range(1, params.timesteps + 1):
            steps = np.sqrt(2 * params.diffusion_coefficient * params.dt) * np.random.randn(params.number_of_particles, 3)
            positions[:, :, t] = positions[:, :, t-1] + steps
            positions[:, 0, t] = np.where(
                positions[:, 0, t] < 0,
                np.abs(positions[:, 0, t]),
                positions[:, 0, t]
            )
            positions[:, 0, t] = np.where(
                positions[:, 0, t] > params.box_size_nm,
                params.box_size_nm - (positions[:, 0, t] - params.box_size_nm),
                positions[:, 0, t]
            )
            positions[:, 1, t] = np.where(
                positions[:, 1, t] < 0,
                np.abs(positions[:, 1, t]),
                positions[:, 1, t]
            )
            positions[:, 1, t] = np.where(
                positions[:, 1, t] > params.box_size_nm,
                params.box_size_nm - (positions[:, 1, t] - params.box_size_nm),
                positions[:, 1, t]
            )
            positions[:, 2, t] = np.where(
                positions[:, 2, t] < 0,
                np.abs(positions[:, 2, t]),
                positions[:, 2, t]
            )
            positions[:, 2, t] = np.where(
                positions[:, 2, t] > params.box_size_nm,
                params.box_size_nm - (positions[:, 2, t] - params.box_size_nm),
                positions[:, 2, t]
            )
        return positions

    @staticmethod
    def simulate_photoswitching(states: np.ndarray, params: SimulationParameters):
        mu_1dt = params.switch_rates[0]*params.dt # F -> Q
        mu_2dt = params.switch_rates[1]*params.dt # Q -> F
        mu_3dt = params.switch_rates[2]*params.dt # F -> H
        mu_4dt = params.switch_rates[3]*params.dt # Q -> H
        mu_5dt = params.switch_rates[4]*params.dt # H -> Z

        for t in range(1, params.timesteps):
            for j in range(params.number_of_particles):
                current_state = states[j, t]
                
                if current_state == 0:
                    r = np.random.rand()
                    if r < mu_1dt:
                        states[j, t+1] = 1
                    elif r < mu_1dt + mu_3dt:
                        states[j, t+1] = 2
                    else:
                        states[j, t] = 0
                if current_state == 1:
                    r = np.random.rand()
                    if r < mu_2dt:
                        states[j, t+1] = 0
                    elif r < mu_2dt + mu_4dt:
                        states[j, t+1] = 2
                    else:
                        states[j, t+1] = 1
                if current_state == 2:
                    r = np.random.rand()
                    if r < mu_5dt:
                        states[j, t+1] = 3
                    else:
                        states[j, t+1] = 2

                elif current_state == 3:
                    states[j, t+1] = 3
        return states

    @staticmethod
    def precompute_normalization(sigma0_pix: float):
        psf_size = int(8 * sigma0_pix)
        if psf_size % 2 == 0:
            psf_size += 1
        center = psf_size // 2
        y, x = np.ogrid[:psf_size, :psf_size]
        gaussian = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma0_pix ** 2))
        return gaussian.sum()

    def psf(self, x_nm: float, y_nm: float, z_nm: float) -> np.ndarray:
        dz = z_nm - self.z_focus_nm
        # Make the PSF less sensitive to z by increasing z_R_nm
        z_R_effective = self.z_R_nm *2.5  # Increase this factor for even less falloff
        sigma_z = self.sigma0_pix * np.sqrt(1 + (dz / z_R_effective) ** 2)
        x_pix = x_nm / self.pixel_size_nm
        y_pix = y_nm / self.pixel_size_nm
        Xc = self.X_grid / self.pixel_size_nm
        Yc = self.Y_grid / self.pixel_size_nm
        psf = np.exp(-((Xc - x_pix) ** 2 + (Yc - y_pix) ** 2) / (2 * sigma_z ** 2))
        psf /= psf.sum()
        return psf

    def render_frame(self, t: int, add_noise: bool = True) -> np.ndarray:
        image = np.zeros((self.params.grid_size, self.params.grid_size), dtype=float)
        for i in range(self.params.number_of_particles):
            state = self.states[i, t]
            if state == 3:
                continue
            intensity = self.params.relative_intensities[state]
            x, y, z = self.positions[i, :, t]
            psf = self.psf(x, y, z)
            particle_brightness = intensity * self.params.expected_photons
            # print(f"Frame {t}, Particle {i}, State {state}, Intensity {intensity}, Brightness {particle_brightness}")
            image += intensity * psf
        image *= self.params.expected_photons  # <-- Only scale, don't normalize by sum
        if add_noise:
            image = np.random.poisson(image).astype(float)
        return image

    def animate(self, interval: int = 50, repeat: bool = False) -> None:
        fig, ax = plt.subplots()
        first_img = self.render_frame(0, add_noise=True)
        vmax = 1500
        im = ax.imshow(
            first_img,
            cmap='inferno',
            vmin=0,
            vmax=vmax,
            animated=True
        )
        time_text = ax.text(
            0.02, 0.95, '', color='w', fontsize=12, transform=ax.transAxes, ha='left', va='top',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )
        ax.set_title("Fluorescence Microscopy Simulation")
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        def update(frame):
            img = self.render_frame(frame, add_noise=True)
            im.set_array(img)
            time_text.set_text(f"Time: {frame * self.params.dt:.2f}")
            return [im, time_text]

        anim = FuncAnimation(
            fig, update, frames=self.params.timesteps + 1,
            interval=interval, blit=True, repeat=repeat
        )
        plt.show()

    def get_particle_trajectories(self) -> np.ndarray:
        return self.positions

    def get_particle_states(self) -> np.ndarray:
        return self.states

    def plot_trajectories(self, n_particles: int = 5) -> None:
        plt.figure(figsize=(6, 6))
        for i in range(min(n_particles, self.params.number_of_particles)):
            plt.plot(self.positions[i, 0, :], self.positions[i, 1, :], label=f'Particle {i+1}')
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.title('Particle Trajectories (2D Projection)')
        plt.legend()
        plt.axis('equal')
        plt.show()

    @staticmethod
    def plot_switching_state_over_time(states: np.ndarray, dt: float, particle_index: int = 0) -> None:
        """
        Plot the switching state of a single particle over time.
        Args:
            states: Array of shape (n_particles, timesteps+1)
            dt: Time step size
            particle_index: Index of the particle to plot
        """
        timesteps = states.shape[1]
        time = np.arange(timesteps) * dt
        plt.figure(figsize=(8, 3))
        plt.step(time, states[particle_index], where='post')
        plt.xlabel("Time")
        plt.ylabel("State")
        plt.title(f"Switching State Over Time (Particle {particle_index})")
        plt.yticks([0, 1, 2, 3], ["F", "Q", "H", "Z"])
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def animate_with_state_trace(self, interval: int = 50, repeat: bool = False, particle_index: int = 0) -> None:
        """
        Animate the fluorescence simulation and the switching state trace side by side.
        """
        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
        ax_img = fig.add_subplot(gs[0])
        ax_trace = fig.add_subplot(gs[1])

        # Prepare image animation
        first_img = self.render_frame(0, add_noise=True)
        vmax = 200
  
        im = ax_img.imshow(
            first_img,
            cmap='inferno',
            vmin=0,
            vmax=vmax,
            animated=True
        )
        time_text = ax_img.text(
            0.02, 0.95, '', color='w', fontsize=12, transform=ax_img.transAxes, ha='left', va='top',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )
        ax_img.set_title("Fluorescence Microscopy Simulation")
        ax_img.axis('off')
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        # Prepare state trace
        timesteps = self.states.shape[1]
        time = np.arange(timesteps) * self.params.dt
        state_line, = ax_trace.step(time, self.states[particle_index], where='post', color='C0')
        marker, = ax_trace.plot([0], [self.states[particle_index, 0]], 'ro', markersize=8)
        ax_trace.set_xlabel("Time")
        ax_trace.set_ylabel("State")
        ax_trace.set_title(f"Switching State Over Time (Particle {particle_index})")
        ax_trace.set_yticks([0, 1, 2, 3])
        ax_trace.set_yticklabels(["F", "Q", "H", "Z"])
        ax_trace.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax_trace.set_xlim(time[0], time[-1])
        ax_trace.set_ylim(-0.5, 3.5)

        def update(frame):
            # Update image
            img = self.render_frame(frame, add_noise=True)
            im.set_array(img)
            time_text.set_text(f"Time: {frame * self.params.dt:.2f}")

            # Update marker on state trace
            marker.set_data([time[frame]], [self.states[particle_index, frame]])
            return [im, time_text, marker]

        anim = FuncAnimation(
            fig, update, frames=self.params.timesteps + 1,
            interval=interval, blit=True, repeat=repeat
        )
        plt.tight_layout()
        plt.show()

    def animate_with_trajectory(
          self,
        interval: int = 50,
        repeat: bool = False,
        particle_index: int = 0,
        show_state_trace: bool = False,
        n_trajectories: int = 1
    ) -> None:
        """
        Animate the fluorescence simulation and the 3D trajectory of up to N particles side by side.
        Optionally, also show the state trace as a third subplot for the main particle_index.
        The trajectory line color reflects the state (F, Q, H, Z) at each segment.
        """

    
        state_colors = {0: 'C0', 1: 'C1', 2: 'C2', 3: 'k'}  # F: blue, Q: orange, H: green, Z: black
    
        if show_state_trace:
            fig = plt.figure(figsize=(16, 5))
            gs = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1])
            ax_img = fig.add_subplot(gs[0])
            ax_traj = fig.add_subplot(gs[1], projection='3d')
            ax_trace = fig.add_subplot(gs[2])
        else:
            fig = plt.figure(figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
            ax_img = fig.add_subplot(gs[0])
            ax_traj = fig.add_subplot(gs[1], projection='3d')
    
        # Prepare image animation
        first_img = self.render_frame(0, add_noise=True)
        vmax = 200
    
        im = ax_img.imshow(
            first_img,
            cmap='inferno',
            vmin=0,
            vmax=vmax,
            animated=True
        )
        time_text = ax_img.text(
            0.02, 0.95, '', color='w', fontsize=12, transform=ax_img.transAxes, ha='left', va='top',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2')
        )
        ax_img.set_title("Fluorescence Microscopy Simulation")
        ax_img.axis('off')
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    
        # Prepare 3D trajectory plot for up to n_trajectories particles
        n_plot = min(n_trajectories, self.params.number_of_particles)
        # For each particle, we will keep a list of line segments (one per state segment)
        traj_lines = []
        for i in range(n_plot):
            # We'll create one line per state segment, but initialize with a single empty line
            line, = ax_traj.plot([], [], [], color='C0', lw=2)
            traj_lines.append([line])  # List of lists for each particle
    
        ax_traj.set_xlabel("X (nm)")
        ax_traj.set_ylabel("Y (nm)")
        ax_traj.set_zlabel("Z (nm)")
        ax_traj.set_title(f"3D Trajectories (up to {n_plot})")
        all_x = self.positions[:n_plot, 0, :].flatten()
        all_y = self.positions[:n_plot, 1, :].flatten()
        all_z = self.positions[:n_plot, 2, :].flatten()
        ax_traj.set_xlim(np.min(all_x), np.max(all_x))
        ax_traj.set_ylim(np.min(all_y), np.max(all_y))
        ax_traj.set_zlim(np.min(all_z), np.max(all_z))
        ax_traj.grid(True, linestyle='--', alpha=0.5)
    
        # Prepare state trace if requested (for particle_index only)
        if show_state_trace:
            timesteps = self.states.shape[1]
            time = np.arange(timesteps) * self.params.dt
            state_line, = ax_trace.step(time, self.states[particle_index], where='post', color='C0')
            marker, = ax_trace.plot([0], [self.states[particle_index, 0]], 'ro', markersize=8)
            ax_trace.set_xlabel("Time")
            ax_trace.set_ylabel("State")
            ax_trace.set_title(f"Switching State Over Time (Particle {particle_index})")
            ax_trace.set_yticks([0, 1, 2, 3])
            ax_trace.set_yticklabels(["F", "Q", "H", "Z"])
            ax_trace.grid(True, axis='x', linestyle='--', alpha=0.5)
            ax_trace.set_xlim(time[0], time[-1])
            ax_trace.set_ylim(-0.5, 3.5)
    
        def update(frame):
            # Update image
            img = self.render_frame(frame, add_noise=True)
            im.set_array(img)
            time_text.set_text(f"Time: {frame * self.params.dt:.2f}")

            # Remove all previous lines from ax_traj
            ax_traj.cla()
            ax_traj.set_xlabel("X (nm)")
            ax_traj.set_ylabel("Y (nm)")
            ax_traj.set_zlabel("Z (nm)")
            ax_traj.set_title(f"3D Trajectories (up to {n_plot})")
            ax_traj.set_xlim(np.min(all_x), np.max(all_x))
            ax_traj.set_ylim(np.min(all_y), np.max(all_y))
            ax_traj.set_zlim(np.min(all_z), np.max(all_z))
            ax_traj.grid(True, linestyle='--', alpha=0.5)
            #ax_traj.view_init(elev=-90, azim=0)  # Look straight down the z-axis onto the x-y plane

            # Draw colored segments for each trajectory up to current frame
            for i in range(n_plot):
                xs = self.positions[i, 0, :frame+1]
                ys = self.positions[i, 1, :frame+1]
                zs = self.positions[i, 2, :frame+1]
                states = self.states[i, :frame+1]
                # Find contiguous segments of the same state
                if len(xs) < 2:
                    continue
                seg_start = 0
                for j in range(1, len(xs)):
                    if states[j] != states[seg_start] or j == len(xs)-1:
                        seg_end = j if states[j] != states[seg_start] else j+1
                        color = state_colors.get(states[seg_start], 'k')
                        ax_traj.plot(xs[seg_start:seg_end], ys[seg_start:seg_end], zs[seg_start:seg_end], color=color, lw=2)
                        seg_start = j
            # Optionally, add a legend for the states
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color=state_colors[0], lw=2, label='F'),
                Line2D([0], [0], color=state_colors[1], lw=2, label='Q'),
                Line2D([0], [0], color=state_colors[2], lw=2, label='H'),
                Line2D([0], [0], color=state_colors[3], lw=2, label='Z'),
            ]
            ax_traj.legend(handles=legend_elements, loc='upper right')

            artists = [im, time_text]
            if show_state_trace:
                marker.set_data([frame * self.params.dt], [self.states[particle_index, frame]])
                artists.append(marker)
            return artists
    
        anim = FuncAnimation(
            fig, update, frames=self.params.timesteps + 1,
            interval=interval, blit=False, repeat=repeat
        )
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    # Example usage: run and animate the simulation
    params = SimulationParameters(
        wavelength=100.0,
        numerical_aperture=1.4,
        refractive_index=1.33,
        box_size_nm=1000.0,
        number_of_particles=100,
        total_time=30.0,  # Use total_time instead of timesteps
        dt=0.1,
        diffusion_coefficient=50.0,
        grid_size=100,
        protein_diameter_nm=4.0,
        expected_photons=10000,
        switch_rates=[0.2, 0.2, 0.05, 0.05, 0.2],  # F→Q, Q→F, F→H, Q→H, H→Z
        relative_intensities=[1, 0.1, 0.5, 0.0]       # F, Q, H, Z
    )
    sim = FluorescenceMicroscopySimulator(params)
    # Plot switching state trace first
    # FluorescenceMicroscopySimulator.plot_switching_state_over_time(
    #     sim.get_particle_states(), params.dt, particle_index=0
    # )
    # Then show the animation
    sim.animate_with_trajectory(
        interval=100, repeat=True, particle_index=0, show_state_trace=True, n_trajectories=params.number_of_particles
    )
    