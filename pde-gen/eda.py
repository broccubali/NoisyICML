from scipy.io import loadmat
import numpy as np
# Load the .mat file
data = loadmat('/home/shusrith/projects/torch/NoisyICML/burgers_shock_mu_01_pi.mat')

# Display the keys in the dictionary
print(data.keys())

# Access specific variables
x = np.array(data['x'])
u = np.array(data['usol'])
print(u.shape, x.shape)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm


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
            im = ax.plot(
                xcrd, data[i].squeeze(), animated=True, color="blue"
            ) 
        ims.append([im[0]])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

    writer = animation.PillowWriter(fps=15, bitrate=1800)
    ani.save(path, writer=writer)
    plt.close(fig)
    
visualize_burgers(x, u.T, "noisy_trained_predicted_solution.gif")