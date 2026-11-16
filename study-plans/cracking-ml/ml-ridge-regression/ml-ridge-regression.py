import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype= np.float64)
    y = np.array(y,dtype=np.float64)
    #y = y.reshape(-1,1)
    w = np.zeros(X.shape[1],dtype=np.float64)
    b = 0.0
    n = X.shape[0]
    for i in range(epochs):
        y_pred = X@w + b
        #y_pred = y_pred.reshape(-1,1)
        y_loss = y_pred - y
        dw = ((2/n) * (X.T @ y_loss)) + 2 * alpha * w
        db = (2/n) * np.sum(y_loss,axis=0)

        w -= lr*dw
        b-= lr*db

    return (w.flatten().tolist(),b)