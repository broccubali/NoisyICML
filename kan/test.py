import torch
from kan import KAN
import h5py
import numpy as np
from visualize import visualize_burgers
from tqdm import tqdm

# Adjust the input dimensions to match the data
model = KAN([1024, 2048, 2048, 2048, 2048, 1024]).to("cuda")

with h5py.File("/home/shusrith/Downloads/simulation_data.hdf", "r") as f:
    l = list(f.keys())[:50]
    d = []
    for i in l:
        d.append([f[i]["clean"][:], f[i]["noisy"][:]])
    d = np.array(d)
    f.close()

noise = torch.Tensor(d[:, 1, :, :] - d[:, 0, :, :])
train = torch.Tensor(d[:, 1, :, :])
visualize_burgers([i for i in range(1024)], noise[0], "a.gif")
# noise = noise.view(-1, 1024).to("cuda")
# train = train.view(-1, 1024).to("cuda")
# loss = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# dataset = torch.utils.data.TensorDataset(train, noise)
# loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.1, patience=10
# )
# for i in range(100):
#     for j in tqdm(loader):
#         inp, out = j
#         optimizer.zero_grad()
#         a = model(inp)
#         l = loss(a, out)
#         l.backward()
#         optimizer.step()
#     scheduler.step(l)
#     print(l)

# a = train[:201].view(-1, 1024)
# print(a.shape)
# a = model(a)
# b = train[:201] - a
# b = b.cpu().detach().numpy()


# print(torch.mean((noise[:201] - a) ** 2))
# print(torch.mean((noise[:201] - train[:201]) ** 2))
