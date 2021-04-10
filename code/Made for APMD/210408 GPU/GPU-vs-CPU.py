# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:22:30 2021

Train a model twice: once on the CPU, once on GPU using CUDA and check the time

@author: ArthurBaucour
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from timeit import default_timer as timer

# =============================================================================
# Getting the data
# =============================================================================

# Loading the data
iris = load_iris()  # Load data from sklearn
X = iris.data
y = iris.target

# Splitting in train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the data to 0 mean and 1 variance
scaler = StandardScaler()
scaler.fit(X_train)     # We fit it on X_train, the data "we can see"
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Enfore the type of the data
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()


# =============================================================================
# Model
# =============================================================================

# Parameters

alpha = 1e-3    # Learning rate
epochs = 1000

# Build the model

# model = torch.nn.Sequential(
#     torch.nn.Linear(4, 50),     # 4 inputs
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 50),
#     torch.nn.ReLU(),
#     torch.nn.Linear(50, 3),     # 3 outputs, one for each possible category
#     torch.nn.Softmax(dim=1)
#     )

model = torch.nn.Sequential(
    torch.nn.Linear(4, 1000),     # 4 inputs
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 3),     # 3 outputs, one for each possible category
    torch.nn.Softmax(dim=1)
    )

# Loss and optimizer

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=0.9)


# =============================================================================
# GPU setup
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

X_train_gpu = X_train.to(device)
X_test_gpu = X_test.to(device)
y_train_gpu = y_train.to(device)
y_test_gpu = y_test.to(device)

# model_gpu = model.to(device)


# =============================================================================
# Classification
# =============================================================================

def train(model, loss_fn, optimizer, X_train, X_test, y_train, y_test,
          epochs=100):

    torch.cuda.synchronize()
    t_start = timer()

    # loss_train_list = []    # Keep track of the training loss
    accuracy_train_list = []
    accuracy_test_list = []

    for i in range(epochs):
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            # Train accuracy
            y_pred = model(X_train)  # Make prediction for the test dataset
            correct = (torch.argmax(y_pred, dim=1) == y_train)
            correct = correct.type(torch.FloatTensor)
            accuracy_train_list.append(correct.mean())

            # Test accuracy
            y_pred = model(X_test)  # Make prediction for the train dataset
            correct = \
                (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
            accuracy_test_list.append(correct.mean())

    torch.cuda.synchronize()
    t_end = timer()

    t = t_end - t_start

    return accuracy_train_list, accuracy_test_list, t


# CPU

print('CPU training...')

accuracy_train_list, accuracy_test_list, t_cpu = \
    train(model, loss_fn, optimizer,
          X_train, X_test, y_train, y_test, epochs)

print('Time:\t{:.3f} (s)'.format(t_cpu))

# GPU

print('GPU training...')

model_gpu = model.to(device)    # Pass the model to the GPU
# Rebuild the optimizer because of new device
optimizer_gpu = torch.optim.SGD(model_gpu.parameters(), lr=alpha, momentum=0.9)

_, _, t_gpu = \
    train(model_gpu, loss_fn, optimizer_gpu,
          X_train_gpu, X_test_gpu, y_train_gpu, y_test_gpu, epochs)

print('Time:\t{:.3f} (s)'.format(t_gpu))


# =============================================================================
# Graph
# =============================================================================

fig1, ax1 = plt.subplots()
ax1.plot(accuracy_train_list, label='Train')
ax1.plot(accuracy_test_list, label='Test')
ax1.set(xlabel='Epoch', ylabel='Accuracy (%)')
ax1.legend()
