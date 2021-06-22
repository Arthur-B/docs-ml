# PyTorch crash course

## Concept

This repository is a crash course in neural networks aimed to teach how to use PyTorch and build enough understanding to apply machine learning to your own projects.
The goal is to get you up and running, show some practical examples of gradual difficulty and let you apply it to your own projects. 

It compiles all the ressources I consumed during my PhD applying neural networks to material science. It is how I would have liked to learn when I started. 

It is NOT an in-depth theoritical course. There are other great online ressources or books regarding the subject. 

## Course architecture

The goal is to start small and slowly introduce new features / more complex code as we go.

| Chapter | Subject | Content |
|---|---|---|
| (1) Regression | (1-1) Linear regression | Manual gradient descent,<br>Implementation in PyTorch,<br>Bonus: plotting training loss |
| | (1-2) General regression | Problem with (1-1) architecture,<br>General regression model,<br>Bonus: introduction to hyperparameter optimization |
| (2) Classification | (2-1) Rule based learning |McCulloch Pitt's neuron model,<br>Rosenblatt's perceptron |
| | (2-2) PyTorch implementation | Linear classification,<br>General classification model |
| | (2-3) Iris dataset | |
| (3) Improving your code | (3-1) Training on GPU | |
| (4) CNN | | |
| (5) Advanced topics | (5-1) GAN | |

## Acknowledgment

I would like to thank @myungjoon who encouraged me to swith from Tensor Flow to PyTorch and helped design the content of the course, members of the @apmd-lab who experienced an early version of the course tailored to our research in nanophotonics and who gave important feedback, and Professor Jonghwa Shin for his support. 

## To-do

* Clean up the rest of the notebooks
* Build a way to display the notebooks online (website?)
* New notebook about GAN?
* Bonus subjects:
    * Split train / test + determine accuracy
    * dealing with dataframes (pandas)
    * Plot in seaborn
