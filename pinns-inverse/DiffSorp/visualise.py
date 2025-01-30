import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm
import h5py
import random

def visualize_burgers(xcrd, data, path):
    """
    This function animates the Burgers equation

    Args:
    path : path to the desired file
    param: PDE parameter of the data shard to be visualized
    """
    fig, ax = plt.subplots()
    ims = []

    for i in tqdm(range(data.shape[0])):
        if i == 0:
            im = ax.plot(xcrd, data[i].squeeze(), animated=True, color="blue")
        else:
            im = ax.plot(xcrd, data[i].squeeze(), animated=True, color="blue")
        ims.append([im[0]])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save(path, writer=writer)
    plt.close(fig)


with h5py.File("diff_sorp.h5", "r") as f:
    for i in range(3):
        l = str(random.randint(0, 1000))
        while l not in f.keys():
            l = str(random.randint(0, 1000))
        d = f[l]["data"][:]
        e = f[l]["noise"][:]
        visualize_burgers([i for i in range(100)], d, f"diff_sorp_{l}.gif")
        visualize_burgers([i for i in range(100)], e, f"diff_sorp_{l}_noise.gif")