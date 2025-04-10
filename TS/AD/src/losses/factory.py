import torch.nn as nn

def create_criterion(loss_name: str, params: dict = {}):
    name = loss_name.lower()

    if name == 'mse':
        criterion = nn.MSELoss(**params)
    elif name == 'mae':
        criterion = nn.L1Loss(**params)
    elif name == 'bce':
        criterion = nn.BCEWithLogitsLoss(**params)
    return criterion