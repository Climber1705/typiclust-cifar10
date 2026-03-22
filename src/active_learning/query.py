import torch
from typing import List
from torch.utils.data import Dataset

from src.active_learning.base import Strategy
from src.active_learning.state import State
from src.active_learning.strategies import RandomStrategy
from src.config import Config
from src.model import build_resnet18_classifier, add_dropout_to_resnet
from src.training import train_classifier


NEEDS_MODEL = {"Uncertainty", "Margin", "Entropy", "DBAL", "BALD", "BADGE"}
NEEDS_DROPOUT = {"DBAL", "BALD"}
NEEDS_COLD_START = NEEDS_MODEL | {"CoreSet"}


def query_strategy(
    name: str,
    strategy: Strategy,
    state: State,
    config: Config,
    train_dataset: Dataset,
    device: torch.device,
) -> List[int]:
    B = config.cluster.B

    if name in NEEDS_COLD_START and len(state.labeled) == 0:
        return RandomStrategy().query(state, B, device)

    if name in NEEDS_MODEL:
        model = build_resnet18_classifier(device=device)

        if name in NEEDS_DROPOUT:
            model = add_dropout_to_resnet(model)

        state.model = train_classifier(
            config=config.evaluation,
            labeled_indices=list(state.labeled),
            train_dataset=train_dataset,
            device=device,
            model_override=model,
        )

    return strategy.query(state, B, device)