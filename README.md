# Raw Neural Network

## Description
In this repository, I built from scratch a library which can create entire Neural Network, using only NumPy library. The main goal of this project was to achieve a comprehensive understanding of the intricacies of neural networks, encompassing diverse elements such as layers, activation functions, loss functions, optimizers, and regularization techniques.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [XOR Example](#xor-example)
  - [Spiral Example](#spiral-example)
  - [Sine Example](#sine-example)
- [Documentation](#documentation)

## Overview
Through this project, users can build complete neural network models, using and customizing different types of layers, activation functions, loss functions, and optimizers according to their needs. The list of all the layers, activation functions, loss functions and optimizers is enumerated in the wiki of this project.

## Installation
To use this toolkit, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.

## Usage
Examples demonstrating the usage of this repository are available in the examples folder.

### XOR Example

To illustrate usage, let's apply it to the XOR problem (a logical operation for exclusive OR).

First, import the neural network components by using the raw_nn import.

```
import raw_nn as nn
```

Then, define the input and desired output as NumPy arrays:

```
X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([[0], [1], [1], [0]])
```

Ensure the input size is shaped as (dataset size, input features), and the output size as (dataset size, output dimension).
In this case, we have 4 elements in our dataset, 2 input features, and one output feature (1 for True and 0 for False).
Thus, the shape of X is (4, 2), and the shape of y is (4, 1).

Instantiate a model by creating an object of the Model class:

```
model = nn.Model()
```

The model provides an add method to append new layers. Create a model with a dense layer with ReLU activation, followed by another dense layer with sigmoid activation:

```
model.add(nn.Dense(2, 10))  # Add a Dense layer with 2 inputs and 10 outputs
model.add(nn.ReLU())
model.add(nn.Dense(10, 1))  # Add a Dense layer with 10 inputs and 1 output
model.add(nn.Sigmoid())
```

Various optimizers are available; to create one, pass the model parameters as arguments and define a custom learning rate. Here, we'll use the SGD optimizer:

```
optimizer = nn.SGD(model.parameters, learning_rate=1e-2)
```

Multiple loss functions are supported. To instantiate one, pass the model as a parameter. Let's use the Binary Cross Entropy:

```
loss_fn = nn.Loss(model)
```

Define 1000 epochs:

```
for epoch in range(1, 1001):
```

For forward propagation, call the model class with the input data:

```
y_pred = model(X)
```

Forward propagation occurs in the order in which the layers were added. Then, compute the loss using loss_fn, passing the predicted output and the true output:

```
loss = loss_fn(y_pred, y)
```

For backward propagation, use the backward method of the loss class:

```
loss_fn.backward()
```

Finally, update the weights of our model with the step method of our optimizer:

```
optimizer.step()
```

Print the loss every 100 epochs to observe the results.

We observe a decreasing loss. Now, if we print the output of the model with the input data, it closely matches our expectations.

The complete file for this example is available in the xor_example file.

### Spiral Example

We'll explore a more complex model by generating data in a spiral pattern requiring classification. We'll generate a training set and a testing set.

Here are the parameters for our data:

```
SAMPLES_PER_CLASS = 1000
CLASSES = 4
NOISE = 0.15
TRAIN_PERCENTAGE = 0.8
```

We created a function to generate the data from these parameters:

```
X, y, X_test, y_test = generate_spiral_data(SAMPLES_PER_CLASS, CLASSES, NOISE, TRAIN_PERCENTAGE)
```

Plotting them gives us:

For the model, we first create a dense layer with 2 inputs and 64 outputs. We add a dropout regularization layer to avoid overfitting before adding a ReLU activation function.
Then, we create a second dense layer with the number of classes as the number of outputs, using a Softmax activation function.

```
model.add(nn.Dense(2, 64))
model.add(nn.ReLU())
model.add(nn.Dropout(0.1))  # Define the dropout rate at 0.1
model.add(nn.Dense(64, CLASSES))
model.add(nn.Softmax())
```

The Model class has train and eval methods to enable or disable dropout layers for training and validation.

```
model.train()  # To enable dropout layers
model.eval()  # To disable dropout layers
```

We also created functions `train_model` and `eval_model` to facilitate model training:

```
from raw_nn.utils.model_utils import train_model, eval_model
```

We need to pass the model and the data as arguments and we can define a custom number of epochs (default is 10) and batch size (default is 32).

```
train_model(model, X, y, X_test, y_test, loss_fn, optimizer, epochs=50, batch_size=16)
```

This function prints the accuracy and the loss for the training set and the testing set at each epoch.

<img width="314" alt="Accuracy and loss" src="https://github.com/lucas-gtr/RawNeuralNetwork/assets/12534925/e77bd22a-aacd-402c-8a4d-d81581e453e7">

It's possible to save the weights of a model or the entire model with all the layers using these functions:

```
model.save_parameters("models/spiral.parameters")  # Save the weights of the model
model.save_model("models/spiral.model")  # Save the model with its layers its weights 
```

Then load a model with:

```
model.load_parameters("models/spiral.parameters")  # Load the parameters of an existing model
model = nn.Model.load_model("models/spiral.model")  # Create an entier model from a file
```

To evaluate a model, use the `eval_model` method:

```
eval_model(model, X_test, y_test, loss_fn)
```

<img width="292" alt="Validation accuracy" src="https://github.com/lucas-gtr/RawNeuralNetwork/assets/12534925/3f702e5e-ad22-4bc3-b3ec-9891554f6b9a">

The complete file for this example is available in the [sprial_example file](src/examples/spiral_example.py).

### Sine example

A last usage examble can be found in the [sine_example file](src/examples/sine_example.py) with a regression model to mimic the sinusoidal function. The model uses Mean Absolute Error as its loss function.

<img width="588" alt="Sine regression model" src="https://github.com/lucas-gtr/RawNeuralNetwork/assets/12534925/5a4f91d0-5147-483e-9a89-0f94655b688e">

## Documentation

A comprehensive documentation for each class of this project (layers, activation functions, optimizers, etc.) is available in the wiki of this project.
