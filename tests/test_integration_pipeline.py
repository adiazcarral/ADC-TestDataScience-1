import torch
import numpy as np
import argparse
from src.adc_testdatascience_1.models.logistic import LogisticRegression  # Corrected import path
from src.adc_testdatascience_1.models.cnn import SimpleCNN  # Assuming correct CNN import
from src.adc_testdatascience_1.models.equivariant_cnn import RotEquivariantCNN  # Assuming correct RotCNN import
from src.adc_testdatascience_1.utils.data_utils import get_dataloaders
from src.adc_testdatascience_1.scripts.test_model import ModelEvaluator  # Assuming correct evaluator class path

def test_full_pipeline(model_name="logistic"):
    # Load data
    _, _, test_loader = get_dataloaders(subset_fraction=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model and path based on model name
    if model_name == "logistic":
        model = LogisticRegression()
        model_path = "src/adc_testdatascience_1/models/logistic.pth"
    elif model_name == "cnn":
        model = SimpleCNN()
        model_path = "src/adc_testdatascience_1/models/cnn.pth"
    elif model_name == "rotcnn":
        model = RotEquivariantCNN()
        model_path = "src/adc_testdatascience_1/models/rotcnn.pth"

    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    evaluator = ModelEvaluator(device=device)
    y_true, y_pred = evaluator.evaluate(model, test_loader, name=model_name)
    
    # Check types and length
    assert isinstance(y_true, np.ndarray), f"Expected numpy.ndarray, got {type(y_true)}"
    assert isinstance(y_pred, np.ndarray), f"Expected numpy.ndarray, got {type(y_pred)}"
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["logistic", "cnn", "rotcnn"], default="logistic"
    )
    args = parser.parse_args()
    
    test_full_pipeline(model_name=args.model)
