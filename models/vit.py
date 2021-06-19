import torch
from torch import nn
from torch import Tensor


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 108, img_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size * patch_size * in_channels, emb_size)
        self.number_of_patches = int(img_size ** 2 / patch_size ** 2)
        self.in_channels = in_channels
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(self.number_of_patches + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(-1, self.number_of_patches, self.patch_size ** 2 * self.in_channels)
        x = self.linear(x)
        batch_size = x.shape[0]
        cls_tokens = torch.cat(batch_size * [self.cls_token])
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 108, n_classes: int = 10):
        super().__init__()
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.layernorm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor):
        out = x[:, 0]
        out = self.layernorm(out)
        out = self.linear(out)
        return out


class ViT(nn.Module):
    def __init__(self, emb_size: int = 108, drop_p: float = 0., forward_expansion: int = 2, nhead: int = 4,
                 activation: str = 'gelu', num_layers: int = 4,
                 in_channels: int = 3, patch_size: int = 4, img_size: int = 32, n_classes: int = 10):
        super().__init__()
        self.transformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=emb_size * forward_expansion,
            dropout=drop_p,
            activation=activation
        )
        self.transformerEncoder = nn.TransformerEncoder(
            encoder_layer=self.transformerEncoderLayer,
            num_layers=num_layers
        )
        self.patch_embeddings = PatchEmbedding(
            in_channels, patch_size, emb_size, img_size
        )
        self.classificationHead = ClassificationHead(emb_size, n_classes)

    def forward(self, x: Tensor):
        out = self.patch_embeddings(x)
        out = self.transformerEncoder(out)
        out = self.classificationHead(out)
        return out
