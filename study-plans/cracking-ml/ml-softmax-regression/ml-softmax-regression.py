import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    X = np.array(X,dtype = np.float64)
    y = np.array(y,dtype=np.int64)

    y = np.eye(num_classes)[y]

    n,d = X.shape

    w = np.zeros((d,num_classes),dtype=np.float64)
    b = np.zeros(num_classes,dtype=np.float64)

    for i in range(n_iters):
        z = X@w + b 
        z_shift = np.exp(z - np.max(z,axis = 1,keepdims=True))
        p = z_shift / (np.sum(z_shift,axis = 1,keepdims=True))

        dw = (1/n) * (X.T @ (p-y))
        db = np.mean((p-y),axis=0)

        w-=lr*dw
        b-=lr*db

    return (w,b)
        
