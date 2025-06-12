import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
timestep = 0.01
Total_time = 100
timevector = np.arange(0, Total_time, timestep)
number_of_states = 4  # F, Q, H, Z

# Rate constants
mu_1 = 0.05  # F -> Q
mu_2 = 0.05  # Q -> F
mu_3 = 0.025  # F -> H
mu_4 = 0.025  # Q -> H
mu_5 = 0.07  # H -> Z

def single_particle_gillespie(initial_state, time_vector):
    """Tracks a single particle's state over time on a regular grid."""
    current_state = initial_state
    current_time = 0.0
    time_index = 0

    state_over_time = np.full(len(time_vector), current_state, dtype=int)

    while current_time < Total_time:
        # Define possible reactions based on current state
        if current_state == 0:  # F
            reactions = [(1, mu_1), (2, mu_3)]  # F -> Q or F -> H
        elif current_state == 1:  # Q
            reactions = [(0, mu_2), (2, mu_4)]  # Q -> F or Q -> H
        elif current_state == 2:  # H
            reactions = [(3, mu_5)]            # H -> Z
        else:  # Z
            break  # terminal state

        propensities = np.array([r[1] for r in reactions])
        total_prop = np.sum(propensities)

        if total_prop == 0:
            state_over_time[time_index:] = current_state
            break

        r1, r2 = np.random.rand(2)
        tau = -np.log(r1) / total_prop
        reaction_index = np.searchsorted(np.cumsum(propensities), r2 * total_prop)

        next_state = reactions[reaction_index][0]

        # Fill in state over regular time grid
        new_time = current_time + tau
        while time_index < len(time_vector) and time_vector[time_index] < new_time:
            state_over_time[time_index] = current_state
            time_index += 1

        # Update state and time
        current_state = next_state
        current_time = new_time

    # Fill any remaining time with final state
    state_over_time[time_index:] = current_state
    return state_over_time

# Run simulation for 1 particle starting in F (state 0)
trajectory = single_particle_gillespie(initial_state=0, time_vector=timevector)

# Plot
plt.figure(figsize=(10, 3))
state_labels = ['F', 'Q', 'H', 'Z']
plt.step(timevector, trajectory, where='post')
plt.yticks(ticks=range(4), labels=state_labels)
plt.xlabel("Time")
plt.ylabel("State")
plt.title("Single Particle State Over Time")
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()