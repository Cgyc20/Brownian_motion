import numpy as np
import matplotlib.pyplot as plt

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

#Example usage 
num = 10
init = np.zeros(num)  # All particles start in the full state
switch_rates = [0.5, 0.5, 0.25, 0.25, 0.7]  # Example switch rates for F->Q, Q->F, F->H, H->Z
time_steps = 100
dt = 0.1

example_states = switching(num, init, switch_rates, time_steps, dt)
times = np.linspace(0, time_steps * dt, time_steps + 1)
# Plotting the states of a particle over time
plt.figure(figsize=(12, 6))
state_labels = ['Full', 'Quenched', 'Half', 'Zero']
plt.step(times, example_states[1], where='post', label='Particle 1', linewidth=2)
plt.yticks(ticks=np.arange(4), labels=state_labels)
plt.xlabel('Time')
plt.ylabel('State')
plt.title('State of Each Particle Over Time')
plt.legend()
plt.show()
