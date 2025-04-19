# src/models/rot_equivariant_cnn.py
from src.adc_testdatascience_1.models.base_model import BaseClassifier
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn


class RotEquivariantCNN(BaseClassifier):
    def __init__(self, num_classes=10):
        super().__init__()
        r2_act = gspaces.Rot2dOnR2(N=8)

        in_type = enn.FieldType(r2_act, [r2_act.trivial_repr])
        self.input_type = in_type

        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, enn.FieldType(r2_act, 8 * [r2_act.regular_repr]), kernel_size=5, padding=2, bias=False),
            enn.ReLU(enn.FieldType(r2_act, 8 * [r2_act.regular_repr]), inplace=True),
            enn.PointwiseMaxPool(enn.FieldType(r2_act, 8 * [r2_act.regular_repr]), kernel_size=2)
        )

        self.block2 = enn.SequentialModule(
            enn.R2Conv(self.block1.out_type, enn.FieldType(r2_act, 16 * [r2_act.regular_repr]), kernel_size=5, padding=2, bias=False),
            enn.ReLU(enn.FieldType(r2_act, 16 * [r2_act.regular_repr]), inplace=True),
            enn.PointwiseMaxPool(enn.FieldType(r2_act, 16 * [r2_act.regular_repr]), kernel_size=2)
        )

        c = self.block2.out_type.size
        self.fc1 = nn.Linear(c * 7 * 7, num_classes)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        return self.fc1(x)