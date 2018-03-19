import numpy as np
import matplotlib.pyplot as plt
from functools import reduce


def generateX(x, polynomial):
    X = []
    for i in range(len(x)):
        row = []
        for j in range(polynomial+1):
            row.append(x[i]**j)
        X.append(row)
    return np.array(X)

def leastSquares(x,t):
    dot_prod = x.T.dot(x)
    inverse = np.linalg.inv(dot_prod)
    w_hat = inverse.dot(x.T).dot(t)
    return w_hat
