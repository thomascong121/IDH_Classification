import torch
import os
from transformers import ViTConfig
# from utils.func import print_msg, select_out_features

from .bridge import FineGrainedPromptTuning, FusionModule
from .side_vit import ViTForImageClassification as SideViT
from .frozen_vit import ViTForImageClassification as FrozenViT
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_terminal_col():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80

def select_out_features(num_classes, criterion):
    out_features = num_classes
    if criterion in ['mean_square_error', 'mean_absolute_error', 'smooth_L1']:
        out_features = 1
    return out_features


def print_msg(msg, appendixs=[], warning=False):
    color = '\033[93m'
    end = '\033[0m'
    print_fn = (lambda x: print(color + x + end)) if warning else print

    max_len = len(max([msg, *appendixs], key=len))
    max_len = min(max_len, get_terminal_col())
    print_fn('=' * max_len)
    print_fn(msg)
    for appendix in appendixs:
        print_fn(appendix)
    print_fn('=' * max_len)

def generate_model(cfg):
    model = build_model(cfg)
    model = model.to(device)

    # the computation of the number of learnable parameters only works when the preloading is disabled
    # if cfg.dataset.preload_path:
    #     frozen_encoder = None
    # else:
    frozen_encoder = build_frozen_encoder(cfg).to(device)

    num_learnable_params = 0
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            num_learnable_params += param.numel()
    if frozen_encoder is not None:
        for _, param in frozen_encoder.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                num_learnable_params += param.numel()

    print('Total params: {}'.format(total_params))
    print('Learnable params: {}'.format(num_learnable_params))
    print('Learnable params ratio: {:.4f}%'.format(num_learnable_params / total_params * 100))

    return frozen_encoder, model


def load_weights(model, checkpoint):
    weights = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(weights, strict=True)
    print_msg('Load weights form {}'.format(checkpoint))    


def build_model(cfg):
    out_features = select_out_features(
        cfg.n_classes,
        'cross_entropy'
    )
    num_layers = len(parse_layers('0-11'))
    vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-384')
    side_dimension = vit_config.hidden_size // 8
    prompts_dim = vit_config.hidden_size // 8
    fusion_module = FusionModule(
        num_layers=num_layers,
        in_dim=vit_config.hidden_size,
        out_dim=side_dimension,
        num_heads=vit_config.num_attention_heads,
        num_prompts=16,
        prompt_dim=prompts_dim,
        prompt_norm=False,
        prompt_proj=True
    )

    side_config = ViTConfig.from_pretrained(
        'google/vit-base-patch16-384',
        num_hidden_layers=num_layers,
        hidden_size=side_dimension,
        intermediate_size=side_dimension * 4,
        image_size=224,
        num_labels=out_features,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0
    )
    side_encoder = SideViT(side_config)

    model = FineGrainedPromptTuning(side_encoder, fusion_module)
    return model


def build_frozen_encoder(cfg):
    frozen_config = ViTConfig.from_pretrained('google/vit-base-patch16-384')
    frozen_config.token_imp = 'global'
    frozen_config.token_ratio = 0.2
    frozen_config.layers_to_extract = parse_layers('0-11')

    frozen_encoder = FrozenViT.from_pretrained(
        'google/vit-base-patch16-384',
        config=frozen_config
    )

    frozen_encoder.eval()
    for p in frozen_encoder.parameters():
        p.requires_grad = False

    return frozen_encoder


def parse_layers(layers_to_extract):
    if '-' in layers_to_extract:
        start, end = layers_to_extract.split('-')
        return list(range(int(start), int(end) + 1))
    elif ',' in layers_to_extract:
        return list(map(int, layers_to_extract.split(',')))
    else:                   
        return [int(layers_to_extract)]