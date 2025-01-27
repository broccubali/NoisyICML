import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

c = 1.0  # Wave speed
x_min, x_max = -1.0, 1.0  # Spatial domain
t_min, t_max = 0.0, 2.0   # Temporal domain
num_x, num_t = 256, 500   # Grid resolution
dx = (x_max - x_min) / (num_x - 1)
dt = 0.9 * dx / c  # CFL condition for stability

# Spatial and temporal grids
x = np.linspace(x_min, x_max, num_x)
t = np.linspace(t_min, t_max, num_t)

u0 = np.exp(-50 * (x**2))  # Initial displacement: Gaussian pulse
u1 = u0.copy()  # Initial velocity: zero
U = np.zeros((num_x, num_t))  # Solution array
U[:, 0] = u0
U[:, 1] = u0 + dt * u1  # First time step using forward Euler

# Time-stepping loop (finite difference)
for n in range(1, num_t - 1):
    U[1:-1, n + 1] = (
        2 * U[1:-1, n] - U[1:-1, n - 1]
        + c**2 * dt**2 / dx**2 * (U[2:, n] - 2 * U[1:-1, n] + U[:-2, n])
    )

np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/wave_solution.npy", U)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/x_coordinate.npy", x)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/t_coordinate.npy", t)

fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Wave Equation Solution Over Time")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, U[:, frame])
    ax.set_title(f"Wave Equation Solution at t = {t[frame]:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=num_t, init_func=init, blit=True)

gif_writer = PillowWriter(fps=20)
ani.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/wave_clean.gif", writer=gif_writer)

plt.close(fig)
print("GIF saved")
