import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import skewnorm


D = 0.5  # Diffusion coefficient
alpha = 0.1  # Sorption coefficient

x_min, x_max = -1.0, 1.0  # Spatial domain
t_min, t_max = 0.0, 2.0   # Temporal domain
num_x, num_t = 256, 500   # Grid resolution

dx = (x_max - x_min) / (num_x - 1)
dt = 0.9 * dx**2 / (2 * D)  # CFL condition for stability

# Spatial and temporal grids
x = np.linspace(x_min, x_max, num_x)
t = np.linspace(t_min, t_max, num_t)

# Initial conditions
u0 = np.exp(-50 * x**2)  # Gaussian pulse
U = np.zeros((num_x, num_t))  # Solution array
U[:, 0] = u0

noise = skewnorm.rvs(a = 1, loc = 0, scale = 0.2, size = U.shape)
U += noise


# First time step using explicit Euler
U[1:-1, 1] = U[1:-1, 0] + dt * (
    D / dx**2 * (U[2:, 0] - 2 * U[1:-1, 0] + U[:-2, 0]) - alpha * U[1:-1, 0]
)

# Time-stepping loop (finite difference)
for n in range(1, num_t - 1):
    diffusion = D * dt / dx**2 * (U[2:, n] - 2 * U[1:-1, n] + U[:-2, n])
    sorption = -alpha * U[1:-1, n] * dt
    U[1:-1, n + 1] = U[1:-1, n] + diffusion + sorption


np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/DiffSorp/diffSorp_solution_noise.npy", U)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/DiffSorp/x_coordinate.npy", x)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/DiffSorp/t_coordinate.npy", t)


fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("DiffSorp Equation Solution Over Time")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, U[:, frame])
    ax.set_title(f"DiffSorp Equation Solution at t = {t[frame]:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=num_t, init_func=init, blit=True)

gif_writer = PillowWriter(fps=20)
ani.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/DiffSorp/diffSorp_noisy.gif", writer=gif_writer)

plt.close(fig)
print("GIF saved")
