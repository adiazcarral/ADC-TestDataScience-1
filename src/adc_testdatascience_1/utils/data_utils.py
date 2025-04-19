import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import os


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

    # Train/Val split
    train_size = int(0.8 * len(subset))
    val_size = len(subset) - train_size
    train_set, val_set = torch.utils.data.random_split(subset, [train_size, val_size])

    # Load test set
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )
