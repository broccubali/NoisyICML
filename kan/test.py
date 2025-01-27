import torch
from kan import KAN
import h5py
import numpy as np
from visualize import visualize_burgers
from tqdm import tqdm

# Adjust the input dimensions to match the data
model = KAN([1024, 512, 256, 128, 64, 128, 256, 512, 1024]).to("cuda")
with h5py.File("/home/shusrith/Downloads/simulation_data.hdf", "r") as f:
    l = list(f.keys())[:10]
    d = []
    for i in l:
        d.append([f[i]["clean"][:], f[i]["noisy"][:]])
    d = np.array(d)
    f.close()

noise = torch.Tensor(d[:, 1, :, :] - d[:, 0, :, :])
train = torch.Tensor(d[:, 1, :, :])
noise = noise.view(-1, 1024).to("cuda")
train = train.view(-1, 1024).to("cuda")
x = torch.stack((train, noise))
print(x.shape)
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loader = torch.utils.data.DataLoader(x, batch_size=100)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)
for i in range(100):
    for j in tqdm(loader):
        inp, out = j
        optimizer.zero_grad()
        a = model(inp)
        l = loss(a, out)
        l.backward()
        optimizer.step()
        print(l.item())
    scheduler.step(l)

a = train[:201].view(-1, 1024)
print(a.shape)
a = model(a)
b = train[:201] - a
b = b.cpu().detach().numpy()


# visualize_burgers([i for i in range(1024)], b, "noisy_trained_predicted_solution.gif")
# visualize_burgers([i for i in range(1024)], train[0], "noisy_solution.gif")
print(torch.mean((noise[:201] - a) ** 2))
print(torch.mean((noise[:201] - train[:201]) ** 2))
