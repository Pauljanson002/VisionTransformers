import torch.optim as optim


def get_optimizer(optimizer_name, parameters, learning_rate, weight_decay=0.01):
    if optimizer_name == 'adamw':
        return optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        return optim.SGD(parameters, lr=learning_rate)
