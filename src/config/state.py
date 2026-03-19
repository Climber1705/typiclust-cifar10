import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Set, Optional
from torch.utils.data import Dataset

@dataclass
class State:
    device: torch.device
    labeled: Set[int]
    dataset: Dataset
    embeddings: np.ndarray
    model: Optional[nn.Module] = None