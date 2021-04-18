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

from celluloid import Camera          # Make gif
from matplotlib.animation import PillowWriter

import seaborn as sns
sns.set()

cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# =============================================================================
# Generate (separable) data
# =============================================================================

# First Gaussian distribution

n1 = 200    # Number of points in first category
x1, y1 = -5, 5              # Mean
x1_std, y1_std = 3, 1.5     # Standard deviation

data1 = np.stack((np.random.normal(x1, x1_std, n1),    # x1
                  np.random.normal(y1, y1_std, n1),    # x2
                  np.ones(n1)), axis=1)                # y, first category is 1

# Second distribution

n2 = 200
x2, y2 = -5, 20     # 10, 10
x2_std, y2_std = 3, 3

data2 = np.stack((np.random.normal(x2, x2_std, n2),
                  np.random.normal(y2, y2_std, n2),
                  np.ones(n2) * -1), axis=1)
# y2 = np.ones(n2) * -1   # Second category is -1

data = np.concatenate((data1, data2), axis=0)   # Concatenate
np.random.shuffle(data)  # Shuffle
X, y = data[:, [0, 1]], data[:, 2]


# Shuffle

# =============================================================================
# Classification
# =============================================================================


def update_weights(X, y, w, alpha):
    """
    Follows equation 1.3 and 1.6 to update the weight vector w.
    """
    w_list_epoch = []    # All the weights for one EPOCH

    for i in range(len(X)):

        # Define the vector according to the textbook
        x_temp = np.array([[1], [X[i, 0]], [X[i, 1]]]).T

        # Combiner output (Eq. 1.3)
        v = np.matmul(x_temp, w.T)

        # Eq 1.6: weight update
        if (v > 0) and (y[i] == -1):
            w -= alpha * x_temp
            w_list_epoch.append(w.squeeze().tolist())

        elif (v <= 0) and (y[i] == 1):
            w += alpha * x_temp
            # w_list_epoch += w.squeeze().tolist()
            w_list_epoch.append(w.squeeze().tolist())

    return w, w_list_epoch


def train(X, y, w, alpha):
    w_list_epoch = 1
    w_list = []
    while w_list_epoch != []:
        w, w_list_epoch = update_weights(X, y, w, alpha)
        w_list += w_list_epoch

    # w_list.pop(-1)
    return w, w_list


# Define the vector according to the textbook
w = np.array([[0.0], [0.0], [0.0]]).T
alpha = 1e-5    # Learning rate

w, w_list = train(X, y, w, alpha)

w = w.squeeze()     # Get rid of unwanted dimensions
# w_list = np.array(w_list)


# =============================================================================
# Graph
# =============================================================================

# Determine the boundary line

x_plot = np.linspace(-20, 20)
y_plot = -(w[1] * x_plot + w[0]) / w[2]

# Make the figure

fig1, ax1 = plt.subplots()

ax1.scatter(data1[:, 0], data1[:, 1], marker='.', alpha=0.3)
ax1.scatter(data2[:, 0], data2[:, 1], marker='.', alpha=0.3)
ax1.plot(x_plot, y_plot, 'k')

ax1.set(xlim=[-20, 20], ylim=[-10, 25], xlabel='x', ylabel='y')

# Make gif

fig2, ax2 = plt.subplots()
camera = Camera(fig2)


# Initialize

def graph_init():
    ax2.scatter(data1[:, 0], data1[:, 1], color=cycle[0],
                marker='.', alpha=0.3)
    ax2.scatter(data2[:, 0], data2[:, 1], color=cycle[1],
                marker='.', alpha=0.3)
    ax2.set(xlim=[-20, 20], ylim=[-10, 25], xlabel='x', ylabel='y')


graph_init()
y_plot2 = 0 * x_plot
ax2.plot(x_plot, y_plot2, 'k')

plt.show()
camera.snap()

# Training phase

for w_temp in w_list:
    graph_init()

    y_plot2 = -(w_temp[1] * x_plot + w_temp[0]) / w_temp[2]
    ax2.plot(x_plot, y_plot2, 'k')

    camera.snap()

# Add frames at the end (3 s)

w_temp = w_list[-1]  # Last weights
y_plot2 = -(w_temp[1] * x_plot + w_temp[0]) / w_temp[2]

for i in range(75):
    graph_init()
    ax2.plot(x_plot, y_plot2, 'k')
    camera.snap()

animation = camera.animate()
writer = PillowWriter(fps=25)

animation.save('test2.gif', writer=writer)
