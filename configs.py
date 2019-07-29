import torch

class BasicConfigs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    embed_size = 64
    num_hiddens = 64
    num_layers = 2
    attention_size = 10
    drop_prob  = 0.5
    lr = 0.01
    num_epochs = 50
