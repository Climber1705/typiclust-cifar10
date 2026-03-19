from abc import ABC, abstractmethod
from typing import List

from src.config import State

class Strategy(ABC):
    @abstractmethod
    def query(self, state: State, budget: int) -> List[int]:
        pass