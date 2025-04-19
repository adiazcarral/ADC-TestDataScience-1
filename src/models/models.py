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


# scripts/train_and_validate.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.data_utils import get_dataloaders
from src.models.logistic import LogisticRegression
from src.models.simple_cnn import SimpleCNN
from src.models.rot_equivariant_cnn import RotEquivariantCNN
from sklearn.metrics import accuracy_score
import argparse


def train_model(model, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):  # keep short for testing
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(targets.cpu().numpy())

        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}, Validation Accuracy: {acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['logistic', 'cnn', 'rotcnn'], default='logistic')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders()

    if args.model == 'logistic':
        model = LogisticRegression()
    elif args.model == 'cnn':
        model = SimpleCNN()
    elif args.model == 'rotcnn':
        model = RotEquivariantCNN()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, val_loader, device)
