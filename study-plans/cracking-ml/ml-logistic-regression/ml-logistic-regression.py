import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X,dtype=np.float64)
    y = np.array(y,dtype=np.float64).reshape(-1,1)

    n = X.shape[0]
    d = X.shape[1]

    w = np.zeros((d,1))
    bias = 0.0
    eps = 1e-15
    
    for i in range(n_iters):
        pred = X@w + bias
        pred = np.clip(pred,-500,500)
        y_pred = (1/(1+np.exp(-pred))).reshape(-1,1)
        dw = (1/n) * (X.T @ (y_pred - y))
        db = np.mean(y_pred-y)
        w -= lr*dw
        bias-= lr*db

    return (w.flatten(),bias)
        