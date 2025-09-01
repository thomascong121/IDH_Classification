from model.ABMIL.model_abmil import Attention, GatedAttention
from model.R2TMIL.model_r2tmil import RRTEncoder
import torch.nn as nn
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def define_abmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    model = Attention(model_size, args.n_classes)
    model = model.to(device)
    return model

def define_r2tmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    size_dict = {"tiny": 192, "ultra_small": 384,
                      "small": 1024, "big": 2048}
    model_params = {
        'mlp_dim': size_dict[model_size],
        'epeg_k':15,
        'crmsa_k':3
    }
    model = RRTEncoder(**model_params).to(device)
    model.size = size_dict[model_size]
    return model
class R2T_ABMIL(nn.Module):
    def __init__(self, args):
        super(R2T_ABMIL, self).__init__()
        self.encoder = define_r2tmil(args)
        self.attention = define_abmil(args)
        self.size = [self.encoder.size]

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        return x
