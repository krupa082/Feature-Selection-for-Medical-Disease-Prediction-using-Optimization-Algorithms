import numpy as np
import pandas as pd

def load_data():
    data = pd.read_csv("dataset.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def random_solution(num_features):
    return np.random.randint(0, 2, num_features)
