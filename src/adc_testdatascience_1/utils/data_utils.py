import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_dataloaders(batch_size=64, data_path='data', seed=42, subset_fraction=1.0, save_subset=True):
    torch.manual_seed(seed)
    random.seed(seed)

    os.makedirs(data_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-180, 180)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full train dataset
    full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)

    if subset_fraction < 1.0:
        print(f"📦 Using a balanced subset of {subset_fraction*100:.0f}% of the training data")

        # Get class-wise indices
        class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(full_dataset):
            class_indices[label].append(idx)

        total_subset_size = int(len(full_dataset) * subset_fraction)
        samples_per_class = total_subset_size // 10

        balanced_indices = []
        for label, indices in class_indices.items():
            selected = random.sample(indices, samples_per_class)
            balanced_indices.extend(selected)

        # Shuffle the selected indices
        random.shuffle(balanced_indices)

        if save_subset:
            torch.save(balanced_indices, os.path.join(data_path, f"subset_indices_{int(subset_fraction*100)}.pt"))
            print(f"💾 Saved subset indices to data/subset_indices_{int(subset_fraction*100)}.pt")

        subset = Subset(full_dataset, balanced_indices)
    else:
        subset = full_dataset

    # Balanced Train/Val split
    indices = subset.indices if isinstance(subset, Subset) else list(range(len(subset)))
    labels = [full_dataset[i][1] for i in indices]
    class_indices = {i: [] for i in range(10)}
    for i, label in zip(indices, labels):
        class_indices[label].append(i)

    train_indices = []
    val_indices = []
    for idxs in class_indices.values():
        random.shuffle(idxs)
        split = int(0.8 * len(idxs))
        train_indices.extend(idxs[:split])
        val_indices.extend(idxs[split:])

    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    # Load test set
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )
