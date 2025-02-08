import numpy as np
import h5py
from scipy.stats import skewnorm

def newData(i, f):
    c = np.random.uniform(0.5, 100)  # Wave speed
    x_min, x_max = -1.0, 1.0  # Spatial domain
    t_min, t_max = 0.0, 2.0   # Temporal domain
    num_x, num_t = 1024, 201   # Grid resolution
    dx = (x_max - x_min) / (num_x - 1)
    dt = 0.9 * dx / c  # CFL condition for stability

    # Spatial and temporal grids
    x = np.linspace(x_min, x_max, num_x)
    t = np.linspace(t_min, t_max, num_t)
    r = np.random.uniform(1, 100)
    u0_clean = np.exp(-r * (x**2))  # Initial displacement: Gaussian pulse
    u1_clean = u0_clean.copy()  # Initial velocity: zero
    U_clean = np.zeros((num_x, num_t))  # Solution array
    U_clean[:, 0] = u0_clean
    U_clean[:, 1] = u0_clean + dt * u1_clean  # First time step using forward Euler

    U_noise = U_clean.copy()

    noise = skewnorm.rvs(a=1, loc=0, scale=0.2, size=U_noise.shape)
    U_noise += noise

    # Time-stepping loop (finite difference)
    for n in range(1, num_t - 1):
        U_clean[1:-1, n + 1] = (
            2 * U_clean[1:-1, n] - U_clean[1:-1, n - 1]
            + c**2 * dt**2 / dx**2 * (U_clean[2:, n] - 2 * U_clean[1:-1, n] + U_clean[:-2, n])
        )

    U_noise = U_clean + noise

    # Remove the singleton dimension
    U_clean = U_clean.squeeze()  # Remove the second dimension (of size 1)
    U_noise = U_noise.squeeze()  # Remove the second dimension (of size 1)
    f.create_group(f"{i}")
    f.create_dataset(f"{i}/clean", data=U_clean)
    f.create_dataset(f"{i}/noisy", data=U_noise)
    f.create_dataset(f"{i}/c", data=c)
    f.create_dataset(f"{i}/init", data=r)

    print(f"Data for c = {c} generated and added to HDF5 file")

# Open the HDF5 file once and write all datasets to it
with h5py.File("data1.h5", 'w') as f:
    for i in range(1, 500):
        newData(i, f)
