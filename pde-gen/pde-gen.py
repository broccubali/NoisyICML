import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from scipy.stats import skewnorm
import scipy

a = loadmat("/home/shusrith/projects/torch/NoisyICML/burgers_shock_IC_sinpi.mat")
x = a["x"]
u = a["usol"].T
t = a["t"]
print(u.shape)


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


data = skewnorm.rvs(a=1, scale=0.2, size=u[0, :].shape)
u[0, :] += data
data = skewnorm.rvs(a=1, scale=0.2, size=u[:, 0].shape)
u[:, 0] += data
data = skewnorm.rvs(a=1, scale=0.2, size=u[:, -1].shape)
u[:, -1] += data
visualize_burgers(x, u, "burgers_shock_IC_sinpi.gif")
d = {"usol": u, "t": t, "x": x}
scipy.io.savemat("noisy.mat", d)
