import torch
from kan import KAN
import h5py
import numpy as np
from visualize import visualize_burgers
from tqdm import tqdm

# Adjust the input dimensions to match the data
model = KAN([201, 512, 512, 1024, 512, 512, 201]).to("cuda")

with h5py.File("simulation_data.h5", "r") as f:
    l = list(f.keys())
    d = []
    for i in l:
        if i != "coords":
            d.append([f[i]["clean"][:], f[i]["noisy"][:]])
    d = np.array(d)
    f.close()

clean = torch.Tensor(d[:, 0, :, :])
train = torch.Tensor(d[:, 1, :, :])
clean, train = clean.permute(0, 2, 1), train.permute(0, 2, 1)
clean = clean.reshape(-1, 201).to("cuda")
train = train.reshape(-1, 201).to("cuda")

loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
dataset = torch.utils.data.TensorDataset(train, clean)
loader = torch.utils.data.DataLoader(dataset, batch_size=3072, shuffle=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10
)
lr = optimizer.param_groups[0]["lr"]
for i in range(1):
    for j in tqdm(loader):
        inp, out = j
        optimizer.zero_grad()
        a = model(inp)
        l = loss(a, out)
        l.backward()
        optimizer.step()
    scheduler.step(l)
    if lr != optimizer.param_groups[0]["lr"]:
        lr = optimizer.param_groups[0]["lr"]
        print("Learning rate changed to", lr)
    print(i, l.item())

a = train[:1024].view(-1, 201)
print(a.shape)
a = model(a)

print(torch.mean((clean[:1024] - a) ** 2))
print(torch.mean((train[:1024] - clean[:1024]) ** 2))
visualize_burgers([i for i in range(1024)], a.cpu().detach().T, "test.gif")
torch.save(model.state_dict(), "model.pth")