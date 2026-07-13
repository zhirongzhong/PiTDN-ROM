"""Initialization utilities: device selection, reproducibility, directory setup."""

import os
import torch
import numpy as np
import random
import warnings


def set_device():
    """Detect and return the available compute device (GPU or CPU).

    Returns:
        Device string ('cuda' or 'cpu').
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device used: ", device)
    torch.set_default_dtype(torch.float64)
    warnings.filterwarnings("ignore")
    return device


def set_reproducibility(HyperParams):
    """Set random seeds for reproducible results across runs.

    Args:
        HyperParams: Configuration object with a `seed` attribute.
    """
    seed = HyperParams.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_path(HyperParams):
    """Create the output directory for network results if it does not exist.

    Args:
        HyperParams: Configuration object with a `net_dir` attribute.
    """
    path = HyperParams.net_dir
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path, exist_ok=False)
