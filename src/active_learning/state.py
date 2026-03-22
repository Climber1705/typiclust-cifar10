import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Set
from torch.utils.data import Dataset

@dataclass
class State:
    labeled: Set[int]
    dataset: Dataset
    embeddings: np.ndarray
    model: Optional[nn.Module] = None