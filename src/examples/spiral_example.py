import numpy as np
import raw_nn as nn
from raw_nn.utils.model_utils import train_model, eval_model
import matplotlib.pyplot as plt


SAMPLES_PER_CLASS = 1000
CLASSES = 4
NOISE = 0.15
TRAIN_PERCENTAGE = 0.8


def generate_spiral_data(num_samples, num_classes, noise_level, train_percentage):
    X = np.zeros((num_samples * num_classes, 2))
    y = np.zeros(num_samples * num_classes, dtype='uint8')
    r = np.linspace(0.0, 2, num_samples)
    for class_number in range(num_classes):
        ix = range(num_samples * class_number, num_samples * (class_number + 1))
        theta = np.linspace(class_number * 4, (class_number + 1) * 4, num_samples)
        # Let's add noise on the angle
        theta = theta + np.random.randn(num_samples) * noise_level
        X[ix] = np.c_[r * np.cos(theta*2), r * np.sin(theta*2)]
        y[ix] = class_number

    shuffle_indices = np.random.permutation(num_samples * num_classes)
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    X = X_shuffled[:int(train_percentage * num_samples * num_classes)]
    X_test = X_shuffled[int(train_percentage * num_samples * num_classes):]

    y = y_shuffled[:int(train_percentage * num_samples * num_classes)]
    y_test = y_shuffled[int(train_percentage * num_samples * num_classes):]

    return X, y, X_test, y_test


X, y, X_test, y_test = generate_spiral_data(SAMPLES_PER_CLASS, CLASSES, NOISE, TRAIN_PERCENTAGE)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# plt.show()

model = nn.Model()

model.add(nn.Dense(2, 64))
model.add(nn.ReLU())
model.add(nn.Dropout(0.1))  # Define the dropout rate at 0.1
model.add(nn.Dense(64, CLASSES))
model.add(nn.Softmax())

# model.load_parameters("models/spiral.parameters")  # Load the parameters of an existing model
# model = nn.Model.load_model("models/spiral.model")  # Create an entier model from a file

optimizer = nn.Adam(model.parameters, learning_rate=1e-2)
loss_fn = nn.CategoricalCrossEntropy(model)

# eval_model(model, X_test, y_test, loss_fn)
train_model(model, X, y, X_test, y_test, loss_fn, optimizer, epochs=10, batch_size=16)

model.save_parameters("models/spiral.parameters")  # Save the weights of the model
model.save_model("models/spiral.model")  # Save the model with its layers its weights
