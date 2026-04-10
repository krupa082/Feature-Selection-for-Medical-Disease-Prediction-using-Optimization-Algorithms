import numpy as np
from utils import load_data, random_solution
from fitness import fitness

X, y = load_data()
num_features = X.shape[1]

best_solution = None
best_score = float('inf')

for _ in range(50):
    solution = random_solution(num_features)
    score = fitness(X, y, solution)

    if score < best_score:
        best_score = score
        best_solution = solution

print("Best Feature Selection:", best_solution)
print("Best Accuracy:", 1 - best_score)
