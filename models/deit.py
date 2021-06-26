import torch
from torch import nn, Tensor
from typing import Union
from models.vit import PatchEmbedding, ClassificationHead
from models.vit_naive import TransformerEncoder


class HardDistillationLoss(nn.Module):
    def __init__(self, teacher: nn.Module, ratio=0.5):
        super().__init__()
        self.teacher = teacher
        self.criterion = nn.CrossEntropyLoss()
        self.ratio = ratio

    def forward(self, inputs: Tensor, outputs: Union[Tensor, Tensor], labels: Tensor) -> Tensor:
        outputs_cls, outputs_dist = outputs
        base_loss = self.criterion(outputs_cls, labels)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion(outputs_dist, teacher_labels)

        return self.ratio * base_loss + (1 - self.ratio) * teacher_loss


class DistillablePatchEmbedding(PatchEmbedding):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 108, img_size: int = 32):
        super().__init__(in_channels, patch_size, emb_size, img_size)
        self.dist_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(self.number_of_patches + 2, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, self.number_of_patches, self.patch_size ** 2 * 3)
        x = self.linear(x)
        batch_size = x.shape[0]
        cls_tokens = torch.cat(batch_size * [self.cls_token])
        dist_tokens = torch.cat(batch_size * [self.dist_token])
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)
        x += self.positions
        return x


class DistillableClassificationHead(ClassificationHead):
    def __init__(self, emb_size: int = 108, n_classes: int = 10):
        super().__init__(emb_size, n_classes)
        self.dist_linear = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        x, x_dist = x[:, 0], x[:, 1]
        x_head = self.linear(x)
        x_dist_head = self.dist_linear(x_dist)
        if self.training:
            out = x_head, x_dist_head
        else:
            out = (x_head + x_dist_head) / 2
        return out


class Deit(nn.Sequential):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 108, img_size: int = 32,
                 depth: int = 7, n_classes: int = 10):
        super().__init__(
            DistillablePatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth),
            DistillableClassificationHead(emb_size, n_classes)
        )

