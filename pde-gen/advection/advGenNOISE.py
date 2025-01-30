import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# Configuration values
save_path = "/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/"
dt_save = 0.01
ini_time = 0.0
fin_time = 2.0
nx = 1024
xL = 0.0
xR = 1.0
beta = 1.0  # Advection velocity
if_show = 1
init_mode = "sin"
noise_level = 0.1  # Noise level for initial condition

# Function to initialize the sine wave with added Gaussian noise
def set_function(x, t, beta, noise_level=noise_level):
    # Generate the sine wave
    u = np.sin(2.0 * np.pi * (x - beta * t))
    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size=u.shape)
    return u + noise

def main():
    print(f"Advection velocity: {beta}")

    # Cell edge coordinates
    xe = np.linspace(xL, xR, nx + 1)
    # Cell center coordinates
    xc = xe[:-1] + 0.5 * (xe[1] - xe[0])
    # Time coordinates
    it_tot = int(np.ceil((fin_time - ini_time) / dt_save)) + 1
    tc = np.arange(it_tot + 1) * dt_save

    # Initial condition (at t = 0) with noise
    u = set_function(xc, t=0, beta=beta)

    # Array to store solutions at each time step
    uu = np.zeros([it_tot, u.shape[0]])
    uu[0] = u

    # Time-stepping loop
    t = ini_time
    i_save = 1
    tm_ini = time.time()

    while t < fin_time:
        print(f"Saving data at t = {t:.3f}")
        u = set_function(xc, t, beta)  # Update u based on advection with noise
        uu[i_save] = u  # Save the solution
        t += dt_save
        i_save += 1

    tm_fin = time.time()
    print(f"Total elapsed time: {tm_fin - tm_ini:.2f} seconds")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "Advection_noisy.npy"), uu)
    np.save(os.path.join(save_path, "x_coordinate.npy"), xe)
    np.save(os.path.join(save_path, "t_coordinate.npy"), tc)

    if if_show:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(xL, xR)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title('Advection Simulation Over Time')

        line, = ax.plot([], [], lw=2)

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(xc, uu[frame])
            ax.set_title(f'Advection at t = {tc[frame]:.2f}')
            return line,

        ani = FuncAnimation(fig, update, frames=it_tot, init_func=init, blit=True)

        gif_writer = PillowWriter(fps=30)
        try:
            ani.save(os.path.join(save_path, "advection_noisy.gif"), writer=gif_writer)
            print("GIF saved successfully!")
        except Exception as e:
            print(f"Error saving GIF: {e}")

        plt.close(fig)

if __name__ == "__main__":
    main()
