from models.vit import ViT
from models.vit_naive import ViTNaive
from models.vgg import VGG
from models.deit import Deit
from models.cct import CCT
from models.ViTLite import ViTLite

def create_model(model_name: str):
    models = ['vit', 'vit_naive', 'vit_lite', 'vgg', 'deit', 'cct','vit_lite_2']
    if model_name not in models:
        raise NotImplementedError("The model you asked is not implemented")
    else:
        if model_name == 'vit_lite':
            return ViTLite(
                num_layers = 7,
                num_heads = 4,
                mlp_ratio = 2,
                embedding_dim= 256,
                patch_size= 4,
            )
        elif model_name =='vit_lite_2':
            return ViTLite(
                num_layers = 14,
                num_heads = 4,
                mlp_ratio = 2,
                embedding_dim= 256,
                patch_size= 4
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
        else:
            return ViT()
