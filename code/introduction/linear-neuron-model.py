# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 11:20:07 2021

@author: ArthurBaucour
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='paper')

# -----------------------------------------------------------------------------
# Make data
# -----------------------------------------------------------------------------

w_ideal = 1
b_ideal = 0

x = np.linspace(-0.6, 1.6)
y_ideal = w_ideal * x + b_ideal

# Regression data

y_noisy = w_ideal * x + b_ideal + np.random.normal(0, 0.25, x.size)

# Classification data

# First Gaussian distribution

n1 = 50    # Number of points in first category
x1, y1 = 0, 1              # Mean
x1_std, y1_std = 0.25, 0.25     # Standard deviation

X1 = np.stack((np.random.normal(x1, x1_std, n1),
               np.random.normal(y1, y1_std, n1)), axis=1)

# Second distribution

n2 = 50
x2, y2 = 1, 0
x2_std, y2_std = 0.25, 0.25

X2 = np.stack((np.random.normal(x2, x2_std, n2),
               np.random.normal(y2, y2_std, n2)), axis=1)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

fig1, axs1 = plt.subplots(1, 3, figsize=(7, 2.5), dpi=300)

axs1[0].plot(x, y_ideal, 'k')

axs1[1].plot(x, y_ideal, 'k')
axs1[1].scatter(x, y_noisy, marker='.', alpha=0.3)
axs1[1].set(title='Regression')


axs1[2].plot(x, y_ideal, 'k')
axs1[2].scatter(X1[:, 0], X1[:, 1], marker='.', alpha=0.3)
axs1[2].scatter(X2[:, 0], X2[:, 1], marker='.', alpha=0.3)
axs1[2].set(title='Classification')

for ax in axs1:
    ax.set(xlabel='x', ylabel='y', xlim=[-0.6, 1.6], ylim=[-0.6, 1.6],
           xticks=[], yticks=[])
    ax.set_aspect('equal')

# fig1.tight_layout()
