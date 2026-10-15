import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a float.
    """
    
    n = len(A)
    A = np.array(A)

    ans = 0
    for i in range(n):
        ans+=A[i][i]

    return ans