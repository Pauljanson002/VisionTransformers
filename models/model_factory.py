from models.vit import ViT
from models.vit_naive import ViTNaive


def create_model(model_name: str):
    models = {
        "vit": ViT,
        "vit_naive": ViTNaive
    }
    return models[model_name]
