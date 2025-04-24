import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from models import LogisticRegression, SimpleCNN, RotEquivariantCNN
from dataloader import get_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameter grids per model
hyperparams = {
    "LogisticRegression": [
        {"lr": 1e-2, "batch_size": 64},
        {"lr": 1e-3, "batch_size": 128},
    ],
    "SimpleCNN": [
        {"lr": 1e-3, "batch_size": 64},
        {"lr": 1e-4, "batch_size": 128},
    ],
    "RotEquivariantCNN": [
        {"lr": 1e-3, "batch_size": 64},
        {"lr": 5e-4, "batch_size": 128},
    ],
}

models = {
    "LogisticRegression": LogisticRegression,
    "SimpleCNN": SimpleCNN,
    "RotEquivariantCNN": RotEquivariantCNN,
}

def train_and_validate(model, train_loader, val_loader, lr, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average="macro")
    return f1, model


def main(model_name):
    assert model_name in models, f"Unsupported model: {model_name}"
    ModelClass = models[model_name]
    grid = hyperparams[model_name]

    best_f1 = 0.0
    best_model = None
    best_config = None

    for config in grid:
        print(f"🔍 Testing config: {config}")
        train_loader, val_loader, _ = get_dataloaders(batch_size=config["batch_size"], subset_fraction=0.1)
        model = ModelClass()
        f1, trained_model = train_and_validate(model, train_loader, val_loader, config["lr"])
        print(f"→ F1 score: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = trained_model
            best_config = config

    # Save best model
    torch.save(best_model.state_dict(), f"{model_name}_best.pth")
    print(f"✅ Best model saved: {model_name}_best.pth")
    print(f"🏆 Best config: {best_config} | F1: {best_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name: LogisticRegression | SimpleCNN | RotEquivariantCNN")
    args = parser.parse_args()
    main(args.model)
