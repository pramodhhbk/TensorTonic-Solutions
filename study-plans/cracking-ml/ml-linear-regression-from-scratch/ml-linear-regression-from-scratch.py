import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X,dtype=np.float64)
    y = np.array(y,dtype=np.float64).reshape(-1,1)
  
    n = X.shape[0]
    d = X.shape[1]
    weight = np.zeros((d,1))
    bias = 0.0

    for i in range(epochs):
        y_pred = X@weight + bias
        y_pred = y_pred.reshape(-1,1)
        loss = (1/n) * np.sum((y_pred - y)**2)
        dw = (2/n)*(X.T @ (y_pred - y))
        db = (2/n) * (np.sum((y_pred - y )))
        weight-=lr*dw
        bias-=lr*db

    return (weight.flatten(),bias)
        