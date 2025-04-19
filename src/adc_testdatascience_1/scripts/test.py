import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from src.adc_testdatascience_1.utils.data_utils import get_dataloaders
from src.adc_testdatascience_1.models.logistic import LogisticRegression
from src.adc_testdatascience_1.models.cnn import SimpleCNN
from src.adc_testdatascience_1.models.equivariant_cnn import RotEquivariantCNN

class ModelEvaluator:
    def __init__(self, device):
        self.device = device
        self.results = {}

    def evaluate(self, model, test_loader, name="Model"):
        model.to(self.device)
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        self.results[name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

        print(f"✅ Evaluation - {name}")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        return self.results[name]

    def plot_confusion_matrix(self, model_name):
        cm = self.results[model_name]['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    def compare_models(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(self.results.keys())
        scores = {metric: [self.results[m][metric] for m in model_names] for metric in metrics}

        x = np.arange(len(model_names))
        width = 0.2

        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, scores[metric], width, label=metric)

        plt.xticks(x + width * (len(metrics) / 2 - 0.5), model_names)
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.title("Model Comparison (Accuracy, Precision, Recall, F1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['logistic', 'cnn', 'rotcnn'], default='logistic')
    args = parser.parse_args()

    _, _, test_loader = get_dataloaders(subset_fraction=0.05)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'logistic':
        model = LogisticRegression()
        model_path = 'models/logistic.pth'
        model_name = 'Logistic'
    elif args.model == 'cnn':
        model = SimpleCNN()
        model_path = 'models/cnn.pth'
        model_name = 'CNN'
    elif args.model == 'rotcnn':
        model = RotEquivariantCNN()
        model_path = 'models/rotcnn.pth'
        model_name = 'RotEquivariantCNN'

    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    evaluator = ModelEvaluator(device=device)
    evaluator.evaluate(model, test_loader, name=model_name)
    evaluator.plot_confusion_matrix(model_name)
