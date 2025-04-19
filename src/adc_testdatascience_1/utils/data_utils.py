# src/utils/data_utils.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random


def get_dataloaders(batch_size=64, data_path='data', seed=42):
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)

    # Balance classes
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(full_dataset):
        class_indices[label].append(idx)

    min_class_size = min(len(indices) for indices in class_indices.values())
    balanced_indices = [random.sample(indices, min_class_size) for indices in class_indices.values()]
    final_indices = [idx for sublist in balanced_indices for idx in sublist]

    subset = Subset(full_dataset, final_indices)

    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_set, val_set = torch.utils.data.random_split(subset, [train_size, val_size])

    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )