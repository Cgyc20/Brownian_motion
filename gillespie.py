import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
timestep = 0.01
Total_time = 10
timevector = np.arange(0, Total_time, timestep)
number_of_states = 4  # F, Q, H, Z

# Rate constants
mu_1 = 0.5  # F -> Q
mu_2 = 0.5  # Q -> F
mu_3 = 0.25  # F -> H
mu_4 = 0.25 # Q -> H
mu_5 = 0.2  # H -> Z

def propensity(state_vector):
    """Calculate propensities given the current state."""
    prop = np.zeros(5)
    prop[0] = mu_1 * state_vector[0]  # F -> Q
    prop[1] = mu_2 * state_vector[1]  # Q -> F
    prop[2] = mu_3 * state_vector[0]  # F -> H
    prop[3] = mu_4 * state_vector[1]  # Q -> H
    prop[4] = mu_5 * state_vector[2]  # H -> Z
    return prop

def gillespie_algorithm(initial_F, time_vector):
    """Run the Gillespie algorithm and output a regular time grid of states."""
    state_vector = np.zeros(number_of_states, dtype=int)
    state_vector[0] = initial_F  # Start with F

    State_matrix = np.zeros((number_of_states, len(time_vector)), dtype=int)
    current_time = 0.0
    time_index = 0

    # Record initial state
    State_matrix[:, 0] = state_vector

    while current_time < Total_time:
        prop = propensity(state_vector)
        total_prop = np.sum(prop)

        if total_prop == 0:
            # Fill rest with current state
            State_matrix[:, time_index:] = state_vector[:, None]
            break

        r1, r2 = np.random.rand(2)
        tau = -np.log(r1) / total_prop
        reaction_index = np.searchsorted(np.cumsum(prop), r2 * total_prop)

        # Update state
        if reaction_index == 0:  # F -> Q
            state_vector[0] -= 1
            state_vector[1] += 1
        elif reaction_index == 1:  # Q -> F
            state_vector[0] += 1
            state_vector[1] -= 1
        elif reaction_index == 2:  # F -> H
            state_vector[0] -= 1
            state_vector[2] += 1
        elif reaction_index == 3:  # Q -> H
            state_vector[1] -= 1
            state_vector[2] += 1
        elif reaction_index == 4:  # H -> Z
            state_vector[2] -= 1
            state_vector[3] += 1

        # Update time
        old_time = current_time
        current_time += tau

        # Fill in State_matrix between old_time and new_time
        while time_index < len(time_vector) and timevector[time_index] < current_time:
            State_matrix[:, time_index] = state_vector
            time_index += 1

    return State_matrix

# Run simulation
initial_F = 1  # Initial number of F molecules

states = np.zeros_like(np.zeros((number_of_states, len(timevector))), dtype=float)

for _ in range(1):
    states += gillespie_algorithm(initial_F, timevector)

states /= 1

# Plot
plt.figure(figsize=(8,5))
labels = ['F', 'Q', 'H', 'Z']
for i in range(4):
    plt.plot(timevector, states[i], label=labels[i])
plt.xlabel('Time')
plt.ylabel('Molecule count')
plt.legend()
plt.tight_layout()
plt.show()