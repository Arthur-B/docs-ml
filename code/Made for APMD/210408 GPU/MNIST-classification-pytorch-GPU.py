# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 16:21:54 2021

https://towardsdatascience.com/pytorch-switching-to-the-gpu-a7c0b21e8a99

@author: ArthurBaucour
"""

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from timeit import default_timer as timer


# =============================================================================
# GPU setup
# =============================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# device = 'cpu'

# =============================================================================
# Getting the data
# =============================================================================

# Getting the data and transforming it properly

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])

train_data = datasets.MNIST(root='data', train=True, download=True,
                            transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True,
                           transform=transform)

# Small dataset so we can just put it in (X,y) without fancy loading

X_train, y_train = train_data.data, train_data.targets
X_test, y_test = test_data.data, test_data.targets

# Ensure proper type and cast to device

X_train, y_train = X_train.float().to(device), y_train.long().to(device)
X_test, y_test = X_test.float().to(device), y_test.long().to(device)

# =============================================================================
# Model
# =============================================================================

# Parameters

alpha = 1e-3    # Learning rate

# Build the model

# model = torch.nn.Sequential(
#     torch.nn.Flatten(),
#     torch.nn.Linear(784, 1024),    # 28*28=784
#     torch.nn.ReLU(),
#     torch.nn.Linear(1024, 512),
#     torch.nn.ReLU(),
#     torch.nn.Linear(512, 256),
#     torch.nn.ReLU(),
#     torch.nn.Linear(256, 128),
#     torch.nn.ReLU(),
#     torch.nn.Linear(128, 10),     # 10 outputs, one for each possible category
#     torch.nn.Softmax(dim=1)
#     ).to(device)                  # Send the model to GPU

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 512),    # 28*28=784
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512, 10),     # 10 outputs, one for each possible category
    torch.nn.Softmax(dim=1)
    ).to(device)                  # Send the model to GPU

# Loss and optimizer

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=0.9)


# =============================================================================
# Classification
# =============================================================================


def train(model, loss_fn, optimizer, X_train, y_train, X_test, y_test,
          epochs=100):

    accuracy_train_list = []
    accuracy_test_list = []

    # Training timer
    torch.cuda.synchronize()
    t_start = timer()

    for i in range(epochs):

        # Make predictions
        y_pred = model(X_train)

        # Measure the accuracy
        total = y_train.size(0)
        correct = \
            (torch.argmax(y_pred, dim=1) == y_train).sum().item()
        train_acc = correct / total * 100.0
        accuracy_train_list.append(train_acc)

        # Loss and adjust the weights
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Test accuracy (no grad)
        with torch.no_grad():
            # Make prediction for the test dataset
            y_pred = model(X_test)
            # Measure the accuracy
            total = y_test.size(0)
            correct = \
                (torch.argmax(y_pred, dim=1) == y_test).sum().item()
            test_acc = correct / total * 100.0
            accuracy_test_list.append(test_acc)

        print('({}) \tTrain acc: {:.2f} (%) | Test acc: {:.2f} (%)'
              .format(i, train_acc, test_acc))

    # Training timer
    torch.cuda.synchronize()
    t_end = timer()
    t = t_end - t_start

    return accuracy_train_list, accuracy_test_list, t


# Training

print('Start training...')

accuracy_train_list, accuracy_test_list, t = \
    train(model, loss_fn, optimizer, X_train, y_train, X_test, y_test,
          epochs=2000)

print('Time:\t{:.3f} (s)'.format(t))


# =============================================================================
# Graph
# =============================================================================

fig1, ax1 = plt.subplots()
ax1.plot(accuracy_train_list, label='Train')
ax1.plot(accuracy_test_list, label='Test')
ax1.set(xlabel='Epoch', ylabel='Accuracy (%)')
ax1.legend()
