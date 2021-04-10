# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:30:49 2021

Solve the XOR classification problem (non-linearly separable)
using a neural network.

@author: ArthurBaucour
"""

import torch
import matplotlib.pyplot as plt

# =============================================================================
# Generate data (XOR problem)
# =============================================================================

# First Gaussian distribution: x=(-1,-1), y=0

n1 = 100    # Number of points in first category
x1, y1 = -10, -10             # Mean
x1_std, y1_std = 0.5, 0.5     # Standard deviation

X1 = torch.normal(mean=torch.ones((n1, 2)) * torch.tensor([x1, y1]),
                  std=torch.tensor([x1_std, y1_std]))

y1 = torch.zeros(n1).long()     # First category is 0

# Second distribution: x=(1,1), y=0

n2 = 100
x2, y2 = 10, 10
x2_std, y2_std = 0.5, 0.5

X2 = torch.normal(mean=torch.ones((n2, 2)) * torch.tensor([x2, y2]),
                  std=torch.tensor([x2_std, y2_std]))

y2 = torch.zeros(n2).long()  # First category is 0

# Third distribution: x=(-1,1), y=1

n3 = 100
x3, y3 = -10, 10
x3_std, y3_std = 0.5, 0.5

X3 = torch.normal(mean=torch.ones((n3, 2)) * torch.tensor([x3, y3]),
                  std=torch.tensor([x3_std, y3_std]))

y3 = torch.ones(n3).long()  # First category is 0

# Fourth distribution: x=(1,-1), y=1

n4 = 100
x4, y4 = 10, -10
x4_std, y4_std = 0.5, 0.5

X4 = torch.normal(mean=torch.ones((n4, 2)) * torch.tensor([x4, y4]),
                  std=torch.tensor([x4_std, y4_std]))

y4 = torch.ones(n4).long()  # First category is 0


# Concatenate our two distributions in one dataset

X = torch.cat((X1, X2, X3, X4), 0)
y = torch.cat((y1, y2, y3, y4), 0)

# =============================================================================
# Classification
# =============================================================================

# Parameters

alpha = 1e-4    # Learning rate
epochs = 20000

# Build the model

model = torch.nn.Sequential(
    torch.nn.Linear(2, 30),     # 2 inputs
    torch.nn.ReLU(),
    torch.nn.Linear(30, 2),     # 2 outputs, one for each possible category
    torch.nn.Softmax(dim=1)
    )

loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=0.9)

for i in range(epochs):
    y_pred = model(X)   # Forward prediction
    loss = loss_fn(y_pred, y)     # Compute the loss

    optimizer.zero_grad()   # Reinitialize the gradients
    loss.backward()
    optimizer.step()

# =============================================================================
# Make predictions
# =============================================================================

n = 100     # Used to make a n by n grid

X1_pred, X2_pred = torch.meshgrid(torch.linspace(-20, 20, n),
                                  torch.linspace(-20, 20, n))

# Shape the tensor properly for prediction
X_pred = torch.stack([X1_pred.reshape(-1), X2_pred.reshape(-1)], 1)

# Predictions
y_pred = model(X_pred)

# Shape again for contourf
y_pred_plot = y_pred[:, 0].reshape((n, n))

# Proper format for contourf

X1_pred = X1_pred.detach().numpy()
X2_pred = X2_pred.detach().numpy()
y_pred_plot = y_pred_plot.detach().numpy()


# =============================================================================
# Graph
# =============================================================================

fig1, ax1 = plt.subplots()

# Scatter plot of first category
ax1.scatter(torch.cat([X1[:, 0], X2[:, 0]]),
            torch.cat([X1[:, 1], X2[:, 1]]), marker='.')

# Scatter plot of second category
ax1.scatter(torch.cat([X3[:, 0], X4[:, 0]]),
            torch.cat([X3[:, 1], X4[:, 1]]), marker='.')

# Contour plot of the probability to belong to class 1
im = ax1.contourf(X1_pred, X2_pred, y_pred_plot,
                  cmap="PuOr", alpha=0.3, vmin=0, vmax=1, zorder=0)
fig1.colorbar(im, ax=ax1)
