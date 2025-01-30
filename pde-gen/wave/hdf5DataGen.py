import numpy as np
import h5py
from scipy.stats import skewnorm

def newData(c, f):
    x_min, x_max = -1.0, 1.0  # Spatial domain
    t_min, t_max = 0.0, 2.0   # Temporal domain
    num_x, num_t = 1024, 201   # Grid resolution
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

    noise = skewnorm.rvs(a=1, loc=0, scale=0.2, size=U_noise.shape)
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

    # Remove the singleton dimension
    U_clean = U_clean.squeeze()  # Remove the second dimension (of size 1)
    U_noise = U_noise.squeeze()  # Remove the second dimension (of size 1)

    # Define the data type for the compound dataset (with fields 'clean' and 'noisy')
    dt = np.dtype([
        ('clean', np.float64, U_clean.shape),  # The shape of the clean data
        ('noisy', np.float64, U_noise.shape)   # The shape of the noisy data
    ])

    # Create a compound dataset for each 'c' with 'clean' and 'noisy' fields
    data = np.zeros(1, dtype=dt)
    data['clean'] = U_clean
    data['noisy'] = U_noise

    # Store the compound dataset in the HDF5 file
    f.create_dataset(f"wave_solution_{c}", data=data)

    print(f"Data for c = {c} generated and added to HDF5 file")

# Open the HDF5 file once and write all datasets to it
with h5py.File("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/wave_solutions_new.h5", 'w') as f:
    for c in range(1, 963):
        newData(c, f)
