import numpy as np
import raw_nn as nn
from raw_nn.utils.model_utils import train_model, eval_model
import matplotlib.pyplot as plt


SAMPLES = 1000
START = -3.14
END = 3.14
TRAIN_PERCENTAGE = 0.8


def generate_sine_data(num_samples, start, end, train_percentage):
    X = np.linspace(start, end, num_samples)
    X = X.reshape(-1, 1)
    y = np.sin(X)

    plt.plot(X, y)
    plt.show()

    shuffle_indices = np.random.permutation(num_samples)
    X_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    X = X_shuffled[:int(train_percentage * num_samples)]
    X_test = X_shuffled[int(train_percentage * num_samples):]

    y = y_shuffled[:int(train_percentage * num_samples)]
    y_test = y_shuffled[int(train_percentage * num_samples):]

    return X, y, X_test, y_test


X, y, X_test, y_test = generate_sine_data(SAMPLES, START, END, TRAIN_PERCENTAGE)

model = nn.Model()

model.add(nn.Dense(1, 64))
model.add(nn.ReLU())
model.add(nn.Dense(64, 1))

model.load_parameters("models/sine.parameters")  # Load the parameters of an existing model
# model = nn.Model.load_model("models/sine.model")  # Create an entier model from a file

optimizer = nn.Adam(model.parameters, learning_rate=1e-3)
loss_fn = nn.MeanSquaredError(model)

# eval_model(model, X_test, y_test, loss_fn)
train_model(model, X, y, X_test, y_test, loss_fn, optimizer, epochs=200, batch_size=16)

model.save_parameters("models/sine.parameters")  # Save the weights of the model
model.save_model("models/sine.model")  # Save the model with its layers its weights

X_predict = np.linspace(-3, 3, 100)
X_predict = X_predict.reshape(-1, 1)

y_predict = model(X_predict)

plt.plot(X_predict, y_predict)
plt.show()
