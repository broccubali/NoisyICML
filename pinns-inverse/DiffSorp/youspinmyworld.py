import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        
        # Define neural network layers
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size) if i % 2 == 0 else nn.Tanh()
                for i in range(20)
            ]
        )
        self.layers.append(nn.Linear(hidden_size, output_size))  # Output layer for C(x, t)

        # Trainable diffusion and sorption coefficients
        self.D = nn.Parameter(torch.tensor([10.0], dtype=torch.float32, requires_grad=True))  # Diffusion coefficient
        self.R = nn.Parameter(torch.tensor([10.0], dtype=torch.float32, requires_grad=True))  # Sorption coefficient

        # Loss function and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        """Feedforward pass to get concentration profile."""
        for layer in self.layers:
            x = layer(x)
        return x

    def residual_loss(self, xtrain):
        """Compute physics-based residual loss for the diffusion-sorption equation."""
        xtrain.requires_grad = True  # autograd for derivatives

        # Predict concentration
        u_pred = self.forward(xtrain)

        # Compute first derivatives
        u_grad = torch.autograd.grad(u_pred, xtrain, torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
        u_x, u_t = u_grad[:, [0]], u_grad[:, [1]]  # ∂C/∂x, ∂C/∂t

        # Compute second derivative (diffusion term)
        u_xx = torch.autograd.grad(u_x, xtrain, torch.ones_like(u_x), create_graph=True)[0][:, [0]]  # ∂²C/∂x²

        # Compute residual
        residual = u_t - (self.D / self.R) * u_xx  # Fixed PDE residual equation

        return self.loss(residual, torch.zeros_like(residual).to(xtrain.device))  # Physics loss

    def total_loss(self, xtrain, utrain):
        """Compute total loss = data loss + physics residual loss."""
        data_loss = self.loss(self.forward(xtrain), utrain)  # Data loss
        residual_loss = self.residual_loss(xtrain)  # Physics-based loss
        return data_loss, residual_loss, data_loss + residual_loss

    def train_model(self, xtrain, utrain, epochs=1000):
        """Train the PINN model."""
        for epoch in tqdm(range(epochs)):
            self.optimizer.zero_grad()
            data_loss, res_loss, total_loss = self.total_loss(xtrain, utrain)
            total_loss.backward()

            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self.optimizer.step()

            if epoch % 1000 == 0:
                print(
                    f"Epoch {epoch}, Loss {total_loss.item():.6f}, "
                    f"Data Loss {data_loss.item():.6f}, Residual Loss {res_loss.item():.6f}, "
                    f"D {self.D.item():.4f}, R {self.R.item():.4f}"
                )


# Load dataset
Nx, Nt = 100, 201

with h5py.File("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pde-gen/diff-sorp/diff_sorp.h5", "r") as f:
    dtrue, rtrue = f["10"]["D"][()], f["10"]["R"][()]
    C_all = f["10"]["data"][:]

# Create input (x, t) and output (C) tensors
x = np.linspace(0, 10, Nx)
t = np.linspace(0, 1, Nt)
X, T = np.meshgrid(x, t)

X_train = torch.tensor(np.hstack((X.flatten()[:, None], T.flatten()[:, None])), dtype=torch.float32).to("cuda")
C_train = torch.tensor(C_all.flatten()[:, None], dtype=torch.float32).to("cuda")

# Randomly select training samples
idx = np.random.choice(Nx * Nt, 10000, replace=False)
xtrain, ctrain = X_train[idx], C_train[idx]

# Train the model
model = PINN(2, 20, 1).to("cuda")
model.train_model(xtrain, ctrain, epochs=10000)

# Evaluate the model
with torch.no_grad():
    C_pred = model.forward(xtrain).cpu().numpy()
    C_pred = C_pred.reshape((Nx, 100))

    print(torch.mean(ctrain.cpu() - C_pred.flatten()))
    plt.plot(C_pred[0], label="Predicted")
    plt.legend()
    plt.show()
