import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple

from src.models import SimCLRModel

def build_embeddings(
    model: SimCLRModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    
    embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Building embeddings"):
            embedding = model.encode(images.to(device)).cpu()

            embeddings.append(embedding)
            all_labels.append(labels)

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(all_labels).numpy()

    return embeddings, labels