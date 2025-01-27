import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

nu = 0.01  # Kinematic viscosity
x_min, x_max = -1.0, 1.0  # Spatial domain
t_min, t_max = 0.0, 1.0   # Temporal domain
num_x, num_t = 256, 500   # Increased resolution for stability
dx = (x_max - x_min) / (num_x - 1)
dt = 0.9 * min(dx / 1.0, dx**2 / nu)  # CFL condition for stability

x = np.linspace(x_min, x_max, num_x)
t = np.linspace(t_min, t_max, int((t_max - t_min) / dt))

# Initial condition: Gaussian pulse
u = np.exp(-50 * (x**2))
U = np.zeros((num_x, len(t)))
U[:, 0] = u

# Time stepping loop (finite difference)
for n in range(1, len(t)):
    u_next = u.copy()
    for i in range(1, num_x - 1):
        u_next[i] = (
            u[i]
            - dt / dx * u[i] * (u[i] - u[i - 1])  # Convection term
            + nu * dt / dx**2 * (u[i + 1] - 2 * u[i] + u[i - 1])  # Diffusion term
        )
    u_next[0] = u_next[-1] = 0  # Dirichlet boundary conditions
    u = u_next.copy()
    U[:, n] = u

# Save the data
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/NavierStokes/navierstokes_solution.npy", U)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/NavierStokes/x_coordinate.npy", x)
np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/NavierStokes/t_coordinate.npy", t)

# Plotting and animation
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], lw=2)
ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Navier-Stokes Equation Solution Over Time")

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x, U[:, frame])
    ax.set_title(f"Navier-Stokes Solution at t = {t[frame]:.2f}")
    return line,

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)

gif_writer = PillowWriter(fps=20)
ani.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/NavierStokes/navierstokes_clean.gif", writer=gif_writer)

plt.close(fig)
print("GIF saved")
