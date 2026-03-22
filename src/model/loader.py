import torch
import torch.nn as nn
from typing import Union
from pathlib import Path

from src.model.simclr import SimCLRModel

def load_model(path: Union[str, Path], device: torch.device) -> SimCLRModel:
    model = SimCLRModel()
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model