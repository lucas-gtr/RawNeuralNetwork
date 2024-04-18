# Raw Neural Network

## Description
In this repository, I built from scratch a library which can create entire Neural Network, using only NumPy library. The main goal of this project was to achieve a comprehensive understanding of the intricacies of neural networks, encompassing diverse elements such as layers, activation functions, loss functions, optimizers, and regularization techniques.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)

## Overview
Through this project, users can build complete neural network models, using and customizing different types of layers, activation functions, loss functions, and optimizers according to their needs. The list of all the layers, activation functions, loss functions and optimizers is enumerated in the wiki of this project.

## Installation
To use this toolkit, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

## Usage

Import the entirety of neural network components by utilizing the raw_nn import.

```
import raw_nn as nn
```

To demonstrate usage, let's apply it to the XOR problem (logical operation for exclusive OR).

Firstly, we need define the input and desired output as NumPy arrays:

```
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])
```

The input size should be shaped as (dataset size, input features), and the output size as (dataset size, output dimension).
Here we have 4 elemnts in our dataset, 2 inputs features and one output feature (1 correpond to True and 0 to False).
So the shape of X is (4, 2) and the shape of y is (4, 1)

To instantiate a model, create an object of the `Model` class:

```
model = nn.Model()
```

The model provides an add method to append new layers. Let's create a model with a dense layer with ReLU activation, followed by another dense layer with sigmoid activation:

```
model.add(nn.Dense(2, 10)) # Add a Dense layer with 2 inputs and 10 outputs
model.add(nn.ReLU())
model.add(nn.Dense(10, 1)) # Add a Dense layer with 10 inputs and 1 output
model.add(nn.Sigmoid())
```

Various optimizers are available; to create one, we just pass the model parameters as arguments. Here, we'll use the `SGD` optimizer:

```
optimizer = nn.SGD(model.parameters)
```

Multiple loss functions are supported. To instantiate one, pass the model as a parameter. Let's utilize the `Binary Cross Entropy`:

```
loss_fn = nn.Loss(model)
```

Let's define 1000 epochs.

For the forward propagation, we can the model class with the input datas:

```
y_pred = model(X)
```

Subsequently, we can compute the loss using `loss_fn`, passing the predicted output and the true output:

```
loss = loss_fn(y_pred, y)
```

Then, for the backward propagation, we can use the backward method of the loss class:

```
loss_fn.backward()
```

Finally, we can update the weights of our model with the `step` method of our optimizer : 

```
optimizer.step()
```

We will print the loss every 100 epochs to observe the results.

<img width="318" alt="XOR loss" src="https://github.com/lucas-gtr/RawNeuralNetwork/assets/12534925/a14afe39-0418-4c99-8dae-39a151d4ee84">

We can see that the loss is decreasing. Now, if we print the outpout of the model with the input datas, we have something very close to what we are expecting : 

<img width="94" alt="XOR results" src="https://github.com/lucas-gtr/RawNeuralNetwork/assets/12534925/48a5d788-3818-4ece-905d-4030a1fcf203">

The complete file of this example is available in the [xor_example file](src/examples/xor_example.py) of [example folder](src/examples).

A complete documentation for each class of this project (layers, activations function, optimizers, etc.) is available in the wiki of this project.



