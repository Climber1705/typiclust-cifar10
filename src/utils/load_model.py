import torch
from src.models import SimCLRModel

def load_model(model_path: str) -> SimCLRModel:
    model = SimCLRModel()
    model.load_state_dict(torch.load(model_path))
    return model