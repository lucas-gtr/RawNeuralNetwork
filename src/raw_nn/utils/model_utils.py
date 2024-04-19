from raw_nn import Model, Loss, CategoricalCrossEntropy, BinaryCrossEntropy, Optimizer
import numpy as np
import math


def train_model(model: Model, X: np.ndarray, y: np.ndarray,
                X_test: np.ndarray, y_test: np.ndarray,
                loss_fn: Loss, optimizer: Optimizer,
                epochs=10, batch_size=32):
    """
   Trains the given model using the provided data.

   Args:
       model: The neural network model to be trained.
       X: The input training data.
       y: The target training labels.
       X_test: The input validation data.
       y_test: The target validation labels.
       loss_fn: The loss function to be used for optimization.
       optimizer: The optimizer for updating model parameters.
       epochs (optional): The number of epochs for training. Defaults to 10.
       batch_size (optional): The batch size for training. Defaults to 64.
   """
    model.train()

    total_train_steps = math.ceil(X.shape[0] / batch_size)

    for epoch in range(epochs):
        print(f'--- Epoch: {epoch + 1} / {epochs} ---')
        model.train()
        correct_predictions = 0
        total_loss = 0
        for step in range(total_train_steps):
            batch_X = X[step * batch_size:(step + 1) * batch_size]
            batch_y = y[step * batch_size:(step + 1) * batch_size]

            batch_y_pred = model(batch_X)
            total_loss += loss_fn(batch_y_pred, batch_y)

            loss_fn.backward()
            optimizer.step()

            if isinstance(loss_fn, BinaryCrossEntropy) or isinstance(loss_fn, CategoricalCrossEntropy):
                predictions = np.argmax(batch_y_pred, axis=1)
                correct_predictions += np.count_nonzero(predictions == batch_y.flatten())

        loss = total_loss / total_train_steps
        if isinstance(loss_fn, BinaryCrossEntropy) or isinstance(loss_fn, CategoricalCrossEntropy):
            acc = correct_predictions / X.shape[0]
            print(f'Train: acc: {acc:.3f}, loss: {loss:.3f}')
        else:
            print(f'Train: loss: {loss:.3f}')
        eval_model(model, X_test, y_test, loss_fn)


def eval_model(model: Model, X_test: np.ndarray, y_test: np.ndarray, loss_fn: Loss):
    """
    Evaluates the model performance on the validation data.

    Args:
        model: The trained neural network model.
        X_test: The input validation data.
        y_test: The target validation labels.
        loss_fn: The loss function used to calculate the loss.
    """
    model.eval()

    y_test_pred = model(X_test)
    total_loss = loss_fn(y_test_pred, y_test)

    if isinstance(loss_fn, BinaryCrossEntropy) or isinstance(loss_fn, CategoricalCrossEntropy):
        predictions = np.argmax(y_test_pred, axis=1)
        acc = np.mean(predictions == y_test.flatten())

        print(f'Validation: acc: {acc:.3f}, ' +
              f'loss: {total_loss:.3f}')
    else:
        print(f'Validation: loss: {total_loss:.3f}')


def predict_model(model: Model, X: np.ndarray):
    """
    Performs predictions using the trained model.

    Args:
        model: The trained neural network model.
        X: The input data for prediction.

    Returns:
        np.ndarray: Predicted labels for the input data.
    """
    model.eval()

    logits = model(X)
    predictions = np.argmax(logits, axis=1)

    return predictions
