from model.CLAM.model_clam import CLAM_SB
from model.R2TMIL.model_r2tmil import RRTEncoder
import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def define_clam(args):
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.mil_method != 'mil':
        if 'ResNet50' in args.ft_model or 'UNI' in args.ft_model:
            model_size = 'small'
        elif 'PLIP' in args.ft_model or 'CONCH' in args.ft_model:
            model_size = 'medium'
        elif 'ViT_S_16' in args.ft_model:
            model_size = 'ultra_small'
        elif 'ViT_T_16' in args.ft_model:
            model_size = 'tiny'
        else:
            raise NotImplementedError
        model_dict.update({"size_arg": model_size})

    if args.subtyping:
        model_dict.update({'subtyping': True})

    if args.B > 0:
        model_dict.update({'k_sample': args.B})

    if args.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes=2)
        instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()

    model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    model.relocate()
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

class R2T_CLAM(nn.Module):
    def __init__(self, args):
        super(R2T_CLAM, self).__init__()
        self.encoder = define_r2tmil(args)
        self.clam = define_clam(args)
        self.size = [self.encoder.size]

    def forward(self, x, label, instance_eval=True, return_features=False, attention_only=False, use_prompt=False):
        x = self.encoder(x)
        x = self.clam(x, label=label, instance_eval=instance_eval, return_features=False, attention_only=False, use_prompt=False)
        return x