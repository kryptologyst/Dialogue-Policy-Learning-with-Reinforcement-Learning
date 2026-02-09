"""Seeding utilities for reproducible experiments."""

import random
import os
from typing import Optional
import numpy as np
import torch


def set_seed(seed: Optional[int] = None) -> int:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed. If None, uses a random seed.
        
    Returns:
        The seed that was set
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for training.
    
    Args:
        device: Device specification ("auto", "cpu", "cuda", "mps")
        
    Returns:
        PyTorch device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device)
