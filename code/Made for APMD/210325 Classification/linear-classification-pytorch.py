# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:44:07 2021

Implement a linear classification in PyTorch using a single neuron without
non-linear activation. We just use softmax activation at the output to
represent the probability to belong in one class or the other.

@author: ArthurBaucour
"""

import torch
import matplotlib.pyplot as plt

# =============================================================================
# Generate data
# =============================================================================

# First Gaussian distribution

n1 = 100    # Number of points in first category
x1, y1 = -5, 5              # Mean
x1_std, y1_std = 3, 1.5     # Standard deviation

X1 = torch.normal(mean=torch.ones((n1, 2)) * torch.tensor([x1, y1]),
                  std=torch.tensor([x1_std, y1_std]))

y1 = torch.zeros(n1).long()     # First category is 0

# Second distribution

n2 = 100
x2, y2 = 5, 10
x2_std, y2_std = 3, 3

X2 = torch.normal(mean=torch.ones((n2, 2)) * torch.tensor([x2, y2]),
                  std=torch.tensor([x2_std, y2_std]))

y2 = torch.ones(n2).long()  # First category is 0

# Concatenate our two distributions in one dataset

X = torch.cat((X1, X2), 0)
y = torch.cat((y1, y2), 0)

# =============================================================================
# Classification
# =============================================================================

# Parameters

alpha = 1e-4
epochs = 20000

# Build the model

model = torch.nn.Sequential(
    torch.nn.Linear(2, 2),  # 2 input (x,y), 2 outputs (category index: 0, 1)
    torch.nn.Softmax(dim=1)      # Softmax activation at the output
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

n = 100

X1_pred, X2_pred = torch.meshgrid(torch.linspace(-20, 20, n),
                                  torch.linspace(-5, 20, n))

# Shape the tensor properly for prediction
X_pred = torch.stack([X1_pred.reshape(-1), X2_pred.reshape(-1)], 1)


# Predictions
y_pred = model(X_pred)

# Shape again for contourf
y_pred_plot = y_pred[:, 0].reshape((n, n))

# # Proper format for contourf

X1_pred = X1_pred.detach().numpy()
X2_pred = X2_pred.detach().numpy()
y_pred_plot = y_pred_plot.detach().numpy()


# =============================================================================
# Graph
# =============================================================================

fig1, ax1 = plt.subplots()

ax1.scatter(X1[:, 0], X1[:, 1], marker='.')
ax1.scatter(X2[:, 0], X2[:, 1], marker='.')

im = ax1.contourf(X1_pred, X2_pred, y_pred_plot,
                  cmap="PuOr", alpha=0.3, vmin=0, vmax=1)
fig1.colorbar(im, ax=ax1)
