import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=100):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for _ in range(epochs):
        linear = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-linear))  # sigmoid

        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)

        weights -= lr * dw
        bias -= lr * db

    return weights, bias
