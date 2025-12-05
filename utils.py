import os
import random
import numpy as np
import torch as T

def set_seeds(seed_value: int) -> None:
    """
    Set the random seeds for Python, NumPy, etc. to ensure
    reproducibility of results.

    Args:
        seed_value (int): The seed value to use for random
            number generation. Must be an integer.

    Returns:
        None
    """
    if isinstance(seed_value, int):
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        T.manual_seed(seed_value)
        T.cuda.manual_seed(seed_value)
        T.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
        T.backends.cudnn.deterministic = True
        T.backends.cudnn.benchmark = False
    else:
        raise ValueError(f"Invalid seed value: {seed_value}. Cannot set seeds.")