import numpy as np
import matplotlib.pyplot as plt

def switching(num_particles, init_state, switch_rates, timesteps):
    """
    Simulates the switching of particles between four states: full, quenched, half, and 
    zero over a number of timesteps."""
    # Parameters:
    # num_particles: Number of particles in the system
    # init_state: Initial state of the particles (array of size num_particles)
    # switch_rates: Array of rates for switching between states [full_rate, quenched_rate, half_rate, zero_rate]
    # timesteps: Number of timesteps to simulate

    #Get information from the input parameters
    init = np.copy(init_state)
    full_rate = switch_rates[0]
    quenched_rate = switch_rates[1]
    half_rate = switch_rates[2]
    zero_rate = switch_rates[3]

    # Initialise arrays to store the state of particles at each timestep
    states = np.zeros((timesteps, num_particles))
    states[0,:] = init

    for i in range(timesteps):
        break

