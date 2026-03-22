from torch.utils.data import DataLoader, Dataset

def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=drop_last,
    )