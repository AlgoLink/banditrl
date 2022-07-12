import numpy as np

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