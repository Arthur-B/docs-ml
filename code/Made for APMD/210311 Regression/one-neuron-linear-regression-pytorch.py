# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:24:51 2021

One neuron linear regression implementation using pytorch
Adapted from pytorch official tutorial:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

@author: ArthurBaucour
"""

import torch
import matplotlib.pyplot as plt

# =============================================================================
# Generate data
# =============================================================================

# y = w*x + b + noise, noise for randomness

w_ideal = 2
b_ideal = -25

x = torch.linspace(0, 100, 1000)
x = x.unsqueeze(-1)     # Adapting the shape [1000] to [1000, 1]

y_ideal = w_ideal * x + b_ideal
y_noisy = w_ideal * x + b_ideal \
            + torch.normal(mean=torch.zeros(x.size()), std=10)

# =============================================================================
# Regression
# =============================================================================

alpha = 1e-3
epochs = 20000

model = torch.nn.Sequential(
    torch.nn.Linear(1, 1)   # 1 input, 1 output
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
