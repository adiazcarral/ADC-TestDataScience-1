import sys
from pathlib import Path

# Add the project root (parent of src/) to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# scripts/train_and_validate.py
import torch
import torch.nn as nn
import torch.optim as optim
from src.adc_testdatascience_1.data.data_utils import get_dataloaders
from models.logistic import LogisticRegression
from models.cnn import SimpleCNN
from models.rot_equivariant_cnn import RotEquivariantCNN
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
