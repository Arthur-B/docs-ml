---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Installation
nav_order: 2
---

# Installation

## Python distribution: Anaconda

<img src="images/Anaconda_Logo_h150.png" alt="Anaconda_Logo"
	title="Anaconda" width="300" height="150" />

First of all, you will need a Python distribution and a development environment. [Anaconda](https://www.anaconda.com/) is a Python distribution for scientific computing that is widely used in Data Science. It comes with most of the useful libraries installed, helps with package management, and provides standard IDE (Spyder and Visual Studio Code). So if you don't know what to use give it a try.

You can get Anaconda Individual Edition [at this adress](https://www.anaconda.com/products/individual).

## Python IDE: Spyder

<img src="images/Spyder_Logo_h150.png" alt="Spyder_Logo"
	title="Spyder" width="150" height="150" />

Then you will need an IDE. Anaconda comes with Spyder and also proposes Visual Studio Code as an alternative. Spyder is a well rounded Python environment aimed at scientists, engineers, and data analysts. In particular, the interface has some similarity with Matlab, which may help if you are used to it.

## Deep-Learning library: PyTorch

<img src="images/PyTorch_Logo_h150.png" alt="PyTorch_Logo"
	title="Spyder" width="605" height="150" />

Finally, we install the library that we are going to use for the tutorials: PyTorch. Also, if you have a NVIDIA GPU, it would be a good time to install CUDA, the parallel computing platform allowing your graphic card to do computations in parallel. This would save a lot of time when training large models.

Please note that PyTorch only supports specific versions of CUDA, so during installation please be careful of which version of CUDA you install.

### GPU acceleration: CUDA

We recommend installing CUDA before PyTorch if you plan on using your GPU at some point.

First, check which versions of CUDA are supported by Pytorch. You can do so at ["Start Locally \| Pytorch"](https://pytorch.org/get-started/locally/). For example, as of April 2021, PyTorch 1.8.1 supports CUDA 10.2 and 11.1.

Then, search for the specific version of CUDA you want (ex: ["CUDA 10.2"](https://developer.nvidia.com/cuda-10.2-download-archive)) and install it.

Do not assume that the latest version of CUDA is the proper one. Please check the compatibility between PyTorch and CUDA.

### PyTorch

To install PyTorch, go to ["Start Locally \| Pytorch"](https://pytorch.org/get-started/locally/) and select the proper options.

### Check the installation

To check that PyTorch and CUDA are installed properly, you can used the following code:

```python
import torch
print(torch.__version__)          # Return the module's version
print(torch.cuda.is_available())  # Return True if CUDA is available
```

## FDTD

Finally, if you are interested into using FDTD by Lumerical to get some simulation data, check with your license manager if you have access to the ["Automation API"](https://www.lumerical.com/products/aapi/).
