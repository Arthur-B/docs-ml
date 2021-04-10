# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:07:06 2021

@author: ArthurBaucour
"""

import torch
from math import pi
import matplotlib.pyplot as plt

# =============================================================================
# Generate data
# =============================================================================

x = torch.linspace(-pi, pi, 1000)
x = x.unsqueeze(-1)     # Adapting the shape [1000] to [1000, 1]

y_ideal = torch.sin(x)
y_noisy = torch.sin(x) + torch.normal(mean=torch.zeros(x.size()), std=0.25)

# =============================================================================
# Regression
# =============================================================================

# Parameters

alpha = 1e-4
epochs = 20000

# Build the model

model = torch.nn.Sequential(
    torch.nn.Linear(1, 10),     # 1 input, 10 hidden nodes
    torch.nn.Sigmoid(),         # Non-linear activation
    torch.nn.Linear(10, 1),     # 10 hidden nods, 1 output
    )

loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.RMSprop(model.parameters(), lr=alpha)

for i in range(epochs):
    y_pred = model(x)   # Forward prediction
    loss = loss_fn(y_pred, y_noisy)     # Compute the loss
    optimizer.zero_grad()   # Reinitialize the gradients
    loss.backward()
    optimizer.step()

y_learned = model(x)
y_learned = y_learned.detach().numpy()

# =============================================================================
# Graph
# =============================================================================

fig1, ax1 = plt.subplots()

ax1.scatter(x, y_noisy, marker='.', alpha=0.3, label='Noisy data')
ax1.plot(x, y_ideal, label='Ideal data')
ax1.plot(x, y_learned, label='Learned parameters')
ax1.legend()
