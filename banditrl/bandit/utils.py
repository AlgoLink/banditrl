import numpy as np
from typing import Union

def check_array(
    array: np.ndarray,
    name: str,
    expected_dim: int = 1,
) -> ValueError:
    """Input validation on an array.
    Parameters
    -------------
    array: object
        Input object to check.
    name: str
        Name of the input array.
    expected_dim: int, default=1
        Expected dimension of the input array.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(
            f"`{name}` must be {expected_dim}D array, but got {type(array)}"
        )
    if array.ndim != expected_dim:
        raise ValueError(
            f"`{name}` must be {expected_dim}D array, but got {array.ndim}D array"
        )
        
def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate sigmoid function."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate softmax function."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator