import random

import numpy as np


def set_seeds(seed: int):
    """
    Sets seeds for all libraries that need them.
    Call this at the start of every run before anything else.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass