import numpy as np
import raw_nn as nn

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y = np.array([0, 1, 1, 0])

model = nn.Model()

model.add(nn.Dense(2, 10))  # Add a Dense layer with 2 inputs and 10 outputs
model.add(nn.ReLU())
model.add(nn.Dense(10, 1))  # Add a Dense layer with 10 inputs and 1 output
model.add(nn.Sigmoid())

optimizer = nn.Adam(model.parameters, learning_rate=1e-2)
loss_fn = nn.BinaryCrossEntropy(model)

for epoch in range(1, 1001):
    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    loss_fn.backward()
    optimizer.step()

    if epoch % 100 == 0:
        test = model(X)
        print(f"epoch {epoch} : {loss}")

np.set_printoptions(suppress=True, precision=4)
print(model(X))
