# src/models/logistic.py
from src.adc_testdatascience_1.models.base_model import BaseClassifier
import torch.nn as nn


class LogisticRegression(BaseClassifier):
    def __init__(self, input_dim=28 * 28, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))