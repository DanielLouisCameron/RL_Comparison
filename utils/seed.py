import random

import numpy as np


def set_seeds(seed):
    """
    Sets seeds for all libraries that need them.
    Call this at the start of every run before anything else.
    """
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
