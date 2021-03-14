# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 11:14:41 2021

Implement a one neuron linear regression.
Follows Burkov, Chap. 4

@author: Arthur Baucour
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# =============================================================================
# Generate data
# =============================================================================

# y = w*x + b + noise, noise for randomness

w_ideal = 2
b_ideal = -25

x = np.linspace(0, 100, 1000)
y_ideal = w_ideal * x + b_ideal
y_noisy = w_ideal * x + b_ideal + np.random.normal(0, 10, x.size)


# =============================================================================
# Regression
# =============================================================================


def update_weights(x, y, w, b, alpha):
    """
    Update the weights, w and b, according to the data x,y and learning rate
    alpha.
    """

    N = len(x)

    # Determine the partial derivatives (eq. 1)

    dl_dw = 0.0
    dl_db = 0.0

    for i in range(N):
        dl_dw += -2 * x[i] * (y[i] - (w*x[i] + b))
        dl_db += -2 * (y[i] - (w*x[i] + b))

    dl_dw /= float(N)
    dl_db /= float(N)

    # Update the weights (eq.2)

    w -= alpha * dl_dw
    b -= alpha * dl_db

    return w, b


def train(x, y, w, b, alpha, epochs):
    for i in range(epochs):
        w, b = update_weights(x, y, w, b, alpha)
        if i % 100 == 0:
            print("EPOCH: ", i)
    return w, b


w, b = 0.0, 0.0
alpha = 1e-4
epochs = 10000

w, b = train(x, y_noisy, w, b, alpha, epochs)

y_learned = w * x + b

# =============================================================================
# Graph
# =============================================================================

fig1, ax1 = plt.subplots()

ax1.scatter(x, y_noisy, marker='.', alpha=0.3, label='Noisy data')
ax1.plot(x, y_ideal, label='Ideal data')
ax1.plot(x, y_learned, label='Learned parameters')
ax1.legend()
ax1.set(xlabel='x', ylabel='y')