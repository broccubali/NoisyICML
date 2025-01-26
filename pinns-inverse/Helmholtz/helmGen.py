import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters for the Helmholtz equation
k = 5.0  # Wavenumber
x_min, x_max = -1.0, 1.0  # Spatial domain
t_min, t_max = 0.0, 1.0   # Temporal domain
num_x, num_t = 256, 100   # Number of points in x and t

# Create a grid of x and t points
x = np.linspace(x_min, x_max, num_x)
t = np.linspace(t_min, t_max, num_t)

# Exact solution for Helmholtz (e.g., a sinusoidal wave)
U_exact = np.sin(k * x[:, None]) * np.cos(k * t[None, :])

# Save the data
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Helmholtz/helmholtz_solution.npy", U_exact)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Helmholtz/x_coordinate.npy", x)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Helmholtz/t_coordinate.npy", t)

# Plot and save as a GIF
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Helmholtz Equation Solution Over Time")

# Initialization function for the animation
def init():
    line.set_data([], [])
    return line,

# Update function for the animation
def update(frame):
    line.set_data(x, U_exact[:, frame])
    ax.set_title(f"Helmholtz Equation Solution at t = {t[frame]:.2f}")
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_t, init_func=init, blit=True)

# Save the animation as a GIF
gif_writer = PillowWriter(fps=20)
ani.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Helmholtz/helmholtz_clean.gif", writer=gif_writer)

plt.close(fig)
print("GIF saved as 'helmholtz_solution.gif'")
