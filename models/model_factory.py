from models.vit import ViT
from models.vit_naive import ViTNaive


def create_model(model_name: str):
    models = ['vit', 'vit_naive', 'vit_lite']
    if model_name not in models:
        raise NotImplementedError("The model you asked is not implemented")
    else:
        if model_name == 'vit_lite':
            return ViTNaive(depth=7, emb_size=256)
        elif model_name == 'vit_naive':
            return ViTNaive()
        else:
            return ViT()
