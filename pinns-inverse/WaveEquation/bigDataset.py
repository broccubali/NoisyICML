import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.stats import skewnorm
import random

def newData(c):
    x_min, x_max = -1.0, 1.0  # Spatial domain
    t_min, t_max = 0.0, 2.0   # Temporal domain
    num_x, num_t = 256, 500   # Grid resolution
    dx = (x_max - x_min) / (num_x - 1)
    dt = 0.9 * dx / c  # CFL condition for stability

    # Spatial and temporal grids
    x = np.linspace(x_min, x_max, num_x)
    t = np.linspace(t_min, t_max, num_t)

    u0_clean = np.exp(-50 * (x**2))  # Initial displacement: Gaussian pulse
    u1_clean = u0_clean.copy()  # Initial velocity: zero
    U_clean = np.zeros((num_x, num_t))  # Solution array
    U_clean[:, 0] = u0_clean
    U_clean[:, 1] = u0_clean + dt * u1_clean  # First time step using forward Euler

    U_noise = U_clean.copy()

    noise = skewnorm.rvs(a = 1, loc = 0, scale = 0.2, size = U_noise.shape)
    U_noise += noise

    # Time-stepping loop (finite difference)
    for n in range(1, num_t - 1):
        U_clean[1:-1, n + 1] = (
            2 * U_clean[1:-1, n] - U_clean[1:-1, n - 1]
            + c**2 * dt**2 / dx**2 * (U_clean[2:, n] - 2 * U_clean[1:-1, n] + U_clean[:-2, n])
        )
    
    for n in range(1, num_t - 1):
        U_noise[1:-1, n + 1] = (
            2 * U_noise[1:-1, n] - U_noise[1:-1, n - 1]
            + c**2 * dt**2 / dx**2 * (U_noise[2:, n] - 2 * U_noise[1:-1, n] + U_noise[:-2, n])
        )

    np.save(f"/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/dataset_wholenum/wave_solution_{c}.npy", U_clean)
    np.save(f"/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/dataset_wholenum/wave_solution_noise_{c}.npy", U_noise)
    np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/dataset_wholenum/x_coordinate.npy", x)
    np.save("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/dataset_wholenum/t_coordinate.npy", t)

for c in range(1, 1001):
    newData(c)
    print(f"Data for c = {c} generated and saved")

