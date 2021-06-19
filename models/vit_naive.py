import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.vit import ClassificationHead, PatchEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 108, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** (-0.5)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, self.num_heads, -1, 3).permute(4, 0, 2, 1, 3)
        queries = qkv[0]
        keys = qkv[1]
        values = qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        # Not learnt
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal , bhlv -> bhav ', att, values)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwarg):
        res = x
        x = self.fn(x, **kwarg)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.1):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 108,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 5, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ViTNaive(nn.Sequential):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 108, img_size: int = 32,
                 depth: int = 5, n_classes: int = 10, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
