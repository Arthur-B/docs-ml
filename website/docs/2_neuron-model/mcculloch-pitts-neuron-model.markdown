---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: "Neuron model"
nav_order: 1
parent: "Artificial neuron"
---

# McCulloch-Pitts neuron model

While neural networks developed really quickly recently due to advances in computer hardware, the idea of an artificial neuron goes back to 1943 where McCulloch and Pitts introduced the concept of neural networks as computing machines. They drew parallels with the behaviour of biological neurons to define a general signal processing model that may be used for learning. The model is presented below.

<img src="images/neuron-model.png" alt="Neuron model"
	title="Artificial neuron mode" width="400" height="200" />

The model first make a linear combinations of the inputs:

<img src="images/equation_v.png" alt="equation v"
  title="Linear combination" width="139" height="13" />

Then, pass it through a non-linear activation function:

<img src="images/equation_output.png" alt="equation output"
  title="Neuron's output" width="159" height="39" />

Throughout the tutorials, we will see that there are a wide range of activation functions that can be used (step function, sigmoid, ReLU, ...). Here, it is [the unit step function](https://en.wikipedia.org/wiki/Heaviside_step_function):

<img src="images/equation_step_function.png" alt="equation step function"
  title="Step function" width="131" height="48" />

This divide the space in two parts, separated by the line v=0.

<img src="images/separate_space.png" alt="separate space"
  title="Separate space" width="400" height="400" />

Therefore, by adjusting the weights and bias, the position of the line can be adjusted to separate two categories.
The question is then:

"How do we learn the proper parameters?"
