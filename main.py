import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

number_of_particles = 10
timesteps = 1000
dt = 0.01  # time step size
diffusion_coefficient = 0.005

X_start, X_end = 0, 1
Y_start = 0
Y_end = 1

positions = np.zeros((number_of_particles, 2, timesteps + 1))
for i in range(number_of_particles):
    positions[i, 0, 0] = np.random.uniform(X_start,X_end)  # x
    positions[i, 1, 0] = np.random.uniform(Y_start,Y_end)  # y

for t in range(1, timesteps + 1):
    steps = np.sqrt(2 * diffusion_coefficient * dt) * np.random.randn(number_of_particles, 2)
    positions[:, :, t] = positions[:, :, t-1] + steps
    positions[:, 0, t] = np.where(positions[:, 0, t] < X_start, np.abs(positions[:, 0, t]), positions[:, 0, t])
    positions[:, 1, t] = np.where(positions[:, 1, t] < Y_start, np.abs(positions[:, 1, t]), positions[:, 1, t])
    positions[:, 0, t] = np.where(positions[:, 0, t] > X_end, X_end - (positions[:, 0, t] - X_end), positions[:, 0, t])
    positions[:, 1, t] = np.where(positions[:, 1, t] > Y_end, Y_end - (positions[:, 1, t] - Y_end), positions[:, 1, t])

fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0, 0], positions[:, 1, 0])
lines = [ax.plot([], [], lw=1)[0] for _ in range(number_of_particles)]
ax.set_xlim(X_start, X_end)
ax.set_ylim(Y_start, Y_end)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Brownian Motion Animation with Trajectories')

def update(frame):
    scat.set_offsets(positions[:, :, frame])
    for i, line in enumerate(lines):
        line.set_data(positions[i, 0, :frame+1], positions[i, 1, :frame+1])
    return [scat] + lines

ani = FuncAnimation(fig, update, frames=timesteps+1, interval=20, blit=True)
plt.show()

#intensity evolution over time
times = np.linspace(0, timesteps * dt, timesteps + 1)
def intensity_simple(pos):
    #pos is an array of positions over time
    intensity = np.zeros(timesteps + 1)
    intensity_close = 1 # intensity at y=0
    for t in range(timesteps + 1):
        intensity[t] = (intensity_close) / (1+pos[0, 1, t]**2)
    return intensity

intensity = intensity_simple(positions)

plt.plot(times, intensity)
plt.xlabel('Time (s)')
plt.ylabel('Intensity')
plt.title('Intensity Evolution Over Time')
plt.xlim(0, timesteps * dt)
plt.show()