from models.vit import ViT
from models.vit_naive import ViTNaive
from models.vgg import VGG
from models.deit import Deit
from models.cct import CCT
from models.ViTLite import ViTLite
from models.DeiTLite import DeiTLite
from models.ViTLiteSeq import ViTLiteSeq


def create_model(model_name: str):
    models = ['vit', 'vit_naive', 'vit_lite', 'vgg', 'deit', 'cct', 'vit_lite_2', 'deit_lite', 'vit_lite_h',
              'vit_lite_seq','vit_lite_100']
    if model_name not in models:
        raise NotImplementedError("The model you asked is not implemented")
    else:
        if model_name == 'vit_lite':
            return ViTLite(
                num_layers=7,
                num_heads=4,
                mlp_ratio=2,
                embedding_dim=256,
                patch_size=4,
            )
        elif model_name == 'vit_lite_100':
            return ViTLite(
                num_layers = 14,
                num_heads=8,
                mlp_ratio= 2,
                embedding_dim=512,
                patch_size=4,
                num_classes = 100
            )
        elif model_name == 'vit_lite_seq':
            return ViTLiteSeq(
                num_layers=14,
                num_heads=4,
                mlp_ratio=2,
                embedding_dim=256,
                patch_size=4
            )
        elif model_name == 'vit_lite_2':
            return ViTLite(
                num_layers=14,
                num_heads=4,
                mlp_ratio=2,
                embedding_dim=256,
                patch_size=4
            )
        elif model_name == 'vit_lite_h':
            return ViTLite(
                num_layers=32,
                num_heads=4,
                mlp_ratio=2,
                embedding_dim=256,
                patch_size=4
            )
        elif model_name == 'vit_naive':
            return ViTNaive()
        elif model_name == 'vgg':
            return VGG('VGG16')
        elif model_name == 'deit':
            return Deit(emb_size=108)
        elif model_name == 'cct':
            return CCT(num_layers=14,
                       num_heads=4,
                       mlp_ratio=2,
                       embedding_dim=256,
                       kernel_size=3,
                       stride=1,
                       padding=1)
        elif model_name == 'deit_lite':
            return DeiTLite(
                num_layers=14,
                num_heads=4,
                mlp_ratio=2,
                embedding_dim=256,
                patch_size=4
            )
        else:
            return ViT()
