import torch
from abc import ABC, abstractmethod
from typing import List

from src.active_learning.state import State

class Strategy(ABC):
    @abstractmethod
    def query(self, state: State, budget: int, device: torch.device) -> List[int]:
        pass