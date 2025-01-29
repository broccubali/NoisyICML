import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList(
            [
                (
                    nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
                    if i % 2 == 0
                    else nn.Tanh()
                )
                for i in range(20)
            ]
        )
        self.layers.append(nn.Linear(hidden_size, output_size))
        self.loss = nn.MSELoss()
        self.beta = nn.Parameter(
            torch.tensor([1], dtype=torch.float32, device="cuda")
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss_fn(self, x, u):
        u_pred = self.forward(x)
        return self.loss(u_pred, u)

    def residual_loss(self, xtrain, fhat):
        x = xtrain[:, 0]
        t = xtrain[:, 1]
        g = xtrain.clone()
        g.requires_grad = True
        u_pred = self.forward(g)
        u_x_t = torch.autograd.grad(
            u_pred,
            g,
            torch.ones([xtrain.shape[0], 1]).to("cuda"),
            retain_graph=True,
            create_graph=True,
        )[0]
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]
        # u_xx = u_xx_tt[:, [0]]
        return self.loss(u_t + (self.beta * u_x), fhat)

    def total_loss(self, xtrain, utrain, fhat):
        a = self.loss_fn(xtrain, utrain)
        b = self.residual_loss(xtrain, fhat)
        return self.loss_fn(xtrain, utrain) + self.residual_loss(xtrain, fhat)

    def train_model(self, xtrain, utrain, epochs=1000):
        fhat = torch.zeros(xtrain.shape[0], 1, device="cuda")
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            loss = self.total_loss(xtrain, utrain, fhat)
            loss.backward()
            self.optimizer.step()
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss {loss.item()}, beta {self.beta.item()}")



u = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/Advection_clean.npy")
x = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/x_coordinate.npy")[:-1]
t = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/t_coordinate.npy")[:-1]
model = PINN(input_size=2, hidden_size=20, output_size=1).to("cuda")
# print(model)

X, T = np.meshgrid(x, t)
xtrue = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
utrue = u.T.flatten()[:, None]  # cuz match dimensions
device = torch.device("cuda")
print(X.shape, T.shape)
idx = np.random.choice(xtrue.shape[0], 10000, replace=False)
print(utrue.shape)
print(idx)
xtrain = xtrue[idx, :]
utrain = utrue[idx]

Xtrain = torch.tensor(xtrain, dtype=torch.float32, device=device)
utrain = torch.tensor(utrain, dtype=torch.float32, device=device)

model = PINN(input_size=2, hidden_size=20, output_size=1).to(device)
model.train_model(Xtrain, utrain, epochs=5000)
