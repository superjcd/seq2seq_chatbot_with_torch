import torch

class BasicConfigs():
    # parameters for overall model training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
