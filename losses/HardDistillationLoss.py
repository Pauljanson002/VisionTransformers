import typing
import torch
import torch.nn as nn
from torch import Tensor


class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module, ratio=0.5):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.CrossEntropyLoss()
        self.ratio = ratio

    def forward(self, inputs: Tensor, outputs: typing.Union[Tensor, Tensor], labels: Tensor) -> Tensor:
        outputs_cls, outputs_dist = outputs
        base_loss = self.criterion(outputs_cls, labels)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion(outputs_dist, teacher_labels)

        return self.ratio * base_loss + (1 - self.ratio) * teacher_loss
