import numpy as np
from gradient_descent import gradient_descent

def fitness(X, y, feature_mask):
    feature_mask = np.round(feature_mask).astype(int)

    if np.sum(feature_mask) == 0:
        return 1

    selected = X[:, feature_mask == 1]

    weights, bias = gradient_descent(selected, y)

    linear = np.dot(selected, weights) + bias
    predictions = (linear > 0).astype(int)

    accuracy = np.mean(predictions == y)

    return 1 - accuracy
