import numpy as np

def make_diagonal(v: list) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, N).
    """
    n = len(v)
    ans = np.zeros((n,n))

    for i in range(n):
        ans[i][i] = v[i]


    return ans
    