import h5py
import numpy as np

with h5py.File("simulation_data.h5", "r+") as f:
    l = list(f.keys())
    e = []
    d = []
    for i in l:
        if i == "coords":
            continue
        try:
            d.append([f[i]["clean"][:], f[i]["noisy"][:]])
            e.append(i)
        except:
            print(i)
    d = np.array(d)
    data = d[:, 0, :, :]
    valid_data_indices = []
    e = np.array(e)
    for i in range(len(data)):
        for j in range(data.shape[1]):
            if any(data[i, j, :] > 2):
                valid_data_indices.append(i)
                break
    print(e[valid_data_indices])
    for i in e[valid_data_indices]:
        del f[i]
    f.close()

