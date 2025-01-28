# import torch
# import torch.nn as nn
# import numpy as np
# from scipy.io import loadmat
# from tqdm import tqdm


# class PINN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(PINN, self).__init__()
#         self.layers = nn.ModuleList(
#             [
#                 (
#                     nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
#                     if i % 2 == 0
#                     else nn.Tanh()
#                 )
#                 for i in range(20)
#             ]
#         )
#         self.layers.append(nn.Linear(hidden_size, output_size))
#         self.loss = nn.MSELoss()
#         self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))  # Initial guess
#         self.D = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))    # Initial guess
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x

#     def loss_fn(self, x, u):
#         u_pred = self.forward(x)
#         return self.loss(u_pred, u)

#     def residual_loss(self, xtrain, fhat):
#         x = xtrain[:, 0] # spatial
#         t = xtrain[:, 1] # temporal
#         g = xtrain.clone()
#         g.requires_grad = True

#         u_pred = self.forward(g) # now I predict u(x,t)

#         u_x_t = torch.autograd.grad(
#             u_pred,
#             g,
#             torch.ones([xtrain.shape[0], 1]).to("cuda"),
#             retain_graph=True,
#             create_graph=True,
#         )[0] # first derivatives wrt x and t

#         u_xx_tt = torch.autograd.grad(
#             u_x_t, g, torch.ones(xtrain.shape).to("cuda"), create_graph=True
#         )[0] # second derivatives wrt x and t

#         u_x = u_x_t[:, [0]]
#         u_t = u_x_t[:, [1]]
#         u_xx = u_xx_tt[:, [0]] # second derivative wrt x
#         # u_tt = u_xx_tt[:, [1]] # second derivative wrt t

#         # Advection-diffusion residual: u_t + beta * u_x = D * u_xx
#         residual = u_t + self.beta * u_x - self.D * u_xx

#         return self.loss(residual, fhat)        
#         # return self.loss(u_t + (self.lambda1 * u_pred * u_x) - (self.lambda2 * u_xx), fhat)

#     def total_loss(self, xtrain, utrain, fhat):
#         return self.loss_fn(xtrain, utrain) + self.residual_loss(xtrain, fhat)

#     def train_model(self, xtrain, utrain, epochs=1000):
#         fhat = torch.zeros(xtrain.shape[0], 1, device="cuda")
#         for epoch in tqdm(range(epochs)):
#             self.optimizer.zero_grad()
#             loss = self.total_loss(xtrain, utrain, fhat)
#             loss.backward()
#             self.optimizer.step()
#             if epoch % 1000 == 0:
#                 print(f"Epoch {epoch}, Loss {loss.item()}, beta, {self.beta.item()}, D, {self.D.item()}")



# model = PINN(input_size=2, hidden_size=20, output_size=1).to("cuda")
# # print(model)

# # u = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/wave_solution.npy")
# # u = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/wave_solution_noise.npy")
# # x = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/x_coordinate.npy")
# # t = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/WaveEquation/t_coordinate.npy")[:-1]

# # X, T = np.meshgrid(x, t)
# # xtrue = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
# # utrue = u.T.flatten()[:, None]  # cuz match dimensions
# # device = torch.device("cuda")

# # idx = np.random.choice(xtrue.shape[0], 10000, replace=False)
# # xtrain = xtrue[idx, :]
# # utrain = utrue[idx]

# u = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/Advection_clean.npy")
# x = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/x_coordinate.npy")
# t = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/t_coordinate.npy")

# # Generate meshgrid for spatial and temporal coordinates
# X, T = np.meshgrid(x, t)

# # Flatten the meshgrid for input to the PINN
# xtrue = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# # Flatten the solution for the corresponding u values
# utrue = u.T.flatten()[:, None]  # Flatten the solution along the time axis

# # Convert to PyTorch tensors
# device = torch.device("cuda")
# Xtrain = torch.tensor(xtrue, dtype=torch.float32, device=device)
# utrain = torch.tensor(utrue, dtype=torch.float32, device=device)

# model = PINN(input_size=2, hidden_size=20, output_size=1).to(device)
# model.train_model(Xtrain, utrain, epochs=5000)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define the PINN model
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

    def train_model(self, Xtrain, utrain, epochs=1000, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(Xtrain)
            loss = criterion(output, utrain)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

# Load the data
u = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/Advection_clean.npy")
x = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/x_coordinate.npy")
t = np.load("/home/pes1ug22am100/Documents/Research and Experimentation/NoisyICML/pinns-inverse/Advection/t_coordinate.npy")

# Check the shape of u before flattening
print(f"Original u shape: {u.shape}")

# Generate meshgrid for spatial and temporal coordinates
X, T = np.meshgrid(x, t)

# Flatten the meshgrid for input to the PINN
xtrue = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

# Reshape u to match the number of samples in xtrue
u = u.T  # Make sure the solution is transposed to match the shape (time_steps, space_points)

# Print the shape of u after transposing
print(f"Reshaped u shape: {u.shape}")

# Flatten u to match the number of samples in xtrue
utrue = u.flatten()[:, None]

# Check the shapes of xtrue and utrue
print(f"xtrue shape: {xtrue.shape}")
print(f"utrue shape: {utrue.shape}")

# Ensure that the number of samples match
if xtrue.shape[0] != utrue.shape[0]:
    # If utrue has extra or missing values, slice or trim it to match xtrue
    min_samples = min(xtrue.shape[0], utrue.shape[0])
    xtrue = xtrue[:min_samples]
    utrue = utrue[:min_samples]

    print(f"Adjusted xtrue shape: {xtrue.shape}")
    print(f"Adjusted utrue shape: {utrue.shape}")

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xtrain = torch.tensor(xtrue, dtype=torch.float32, device=device)
utrain = torch.tensor(utrue, dtype=torch.float32, device=device)

# Model initialization and training
model = PINN(input_size=2, hidden_size=20, output_size=1).to(device)
model.train_model(Xtrain, utrain, epochs=5000)
