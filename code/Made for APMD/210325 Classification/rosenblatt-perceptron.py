# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:32:52 2021

Implement the Rosenblatt's perceptron as described in Haykin, Chap. 1
We tried to keep the same structure ("update weight", "train") as used in the
regression section.

@author: ArthurBaucour
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Generate (separable) data
# =============================================================================

# First Gaussian distribution

n1 = 200    # Number of points in first category
x1, y1 = -5, 5              # Mean
x1_std, y1_std = 3, 1.5     # Standard deviation

X1 = np.stack((np.random.normal(x1, x1_std, n1),
               np.random.normal(y1, y1_std, n1)), axis=1)
y1 = np.ones(n1)    # First category is 1

# Second distribution

n2 = 200
x2, y2 = 10, 10
x2_std, y2_std = 3, 3

X2 = np.stack((np.random.normal(x2, x2_std, n2),
               np.random.normal(y2, y2_std, n2)), axis=1)
y2 = np.ones(n2) * -1   # Second category is -1

X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((y1, y2), axis=0)

# =============================================================================
# Classification
# =============================================================================


def update_weights(X, y, w, alpha):
    """
    Follows equation 1.3 and 1.6 to update the weight vector w.
    """

    for i in range(len(X)):
        # Define the vector according to the textbook
        x_temp = np.array([[1], [X[i, 0]], [X[i, 1]]]).T

        # Combiner output (Eq. 1.3)
        v = np.matmul(x_temp, w.T)

        # Eq 1.6: weight update
        if (v > 0) and (y[i] == -1):
            w -= alpha * x_temp
        elif (v <= 0) and (y[i] == 1):
            w += alpha * x_temp
    return w


def train(X, y, w, alpha, epochs):
    for i in range(epochs):
        w = update_weights(X, y, w, alpha)
        if i % 1000 == 0:
            print("EPOCH: ", i)
    return w


# Define the vector according to the textbook
w = np.array([[0.0], [0.0], [0.0]]).T
alpha = 1e-4    # Learning rate
epochs = 1000

w = train(X, y, w, alpha, epochs)
w = w.squeeze()     # Get rid of unwanted dimensions


# =============================================================================
# Graph
# =============================================================================

# Determine the boundary line

x_plot = np.linspace(-10, 10)
y_plot = -(w[1] * x_plot + w[0]) / w[2]

# Make the figure

fig1, ax1 = plt.subplots()

ax1.scatter(X1[:, 0], X1[:, 1], marker='.', alpha=0.3)
ax1.scatter(X2[:, 0], X2[:, 1], marker='.', alpha=0.3)
ax1.plot(x_plot, y_plot, 'k')

ax1.set(xlim=[-20, 20], ylim=[-10, 25])
