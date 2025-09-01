import inspect

import torch
import sys
import torchvision
from transformers import CLIPVisionModelWithProjection
sys.path.insert(1, '/scratch/sz65/cc0395/WSI_prompt/')
from model.FExtractor.resnet import resnet50_baseline, resnet18_baseline
from model.FExtractor.resnet_prompt import PromptResNet
from model.FExtractor.vit_prompt import ViT
from model.FPT.builder import generate_model, load_weights
# from model.FExtractor.vit_adp import ADPTTransformer
# from utils.core_util import replace_bn_layers, replace_ln_layers, custom_load_pretrained, load_pretrained_vit
from utils.core_util import replace_bn_layers, custom_load_pretrained
class PLIP(torch.nn.Module):
    def __init__(self):
        super(PLIP,self).__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained("vinid/plip")
    def forward(self, input):
        return self.model(input).image_embeds

def get_extractor(args, pretrained=True):
    model_name = args.ft_model
    custom_pretrained = args.custom_pretrained
    transfer_type = args.transfer_type
    print(f"Using model: {model_name}")
    if 'ResNet50' in model_name:
        extra_layer = True if 'big' in model_name else False
        model = resnet50_baseline(pretrained=pretrained, extra_block=extra_layer)
        if 'simclr' in model_name:
            model = custom_load_pretrained(model, model_name, custom_pretrained)
        if 'prompt' in model_name:
            model = PromptResNet(args, pretrained_model=model)
    elif 'ResNet18' in model_name:
        extra_layer = True if 'big' in model_name else False
        model = resnet18_baseline(pretrained=pretrained, extra_block=extra_layer)
        if 'simclr' in model_name:
            model = custom_load_pretrained(model, model_name, custom_pretrained)
        if 'prompt' in model_name:
            model = PromptResNet(args, pretrained_model=model)
    elif 'PLIP' in model_name:
        model = PLIP()
    elif 'FPT' in model_name:
        frozen_encoder, side_model = generate_model(args)
        model = (frozen_encoder, side_model)
    elif 'ViT' in model_name:
        if 'ViT_S_16'in model_name:
            n_token = 0
            vit_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=3, global_pool='avg', num_classes=args.n_classes)
            variant = "vit_small_patch16_224"
        elif 'ViT_T_16' in model_name:
            n_token = 0
            vit_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, global_pool='avg', num_classes=args.n_classes)
            variant = "vit_tiny_patch16_224"
        else:
            raise NotImplementedError
        model = ViT(args, vit_kwargs, variant, pretrained)
        # if 'prompt' in transfer_type:
        #     n_token = n_prompt
        #     model = PromptedTransformer(vit_kwargs, n_token, 0.,
        #                                 deep_prompt=False, deep_feature_prompt=False,
        #                                 project_prompt_dim=-1, prompt_aggregation=args.prompt_aggregation)
        #     model = load_pretrained_vit(model, model_name, variant, vit_kwargs, custom_pretrained,
        #                                 pretrained=pretrained)
        # elif transfer_type == 'Adapter':
        #     adpt_kwargs = dict(STYLE="Pfeiffer", REDUCATION_FACTOR=args.adpt_rf)
        #     model = ADPTTransformer(vit_kwargs, variant, adpt_kwargs)
        # else:
        #     model = BaseTransformer(vit_kwargs)
        #     model = load_pretrained_vit(model, model_name, variant, vit_kwargs, custom_pretrained,
        #                                 pretrained=pretrained)
        # if args.deep_prompt:
        #     print("Using PromptedTransformer with deep prompt")
        #     model = PromptedTransformer(vit_kwargs, n_token, 0.,
        #                                 deep_prompt=True, deep_feature_prompt=False,
        #                                 project_prompt_dim=-1, prompt_aggregation=args.prompt_aggregation)
        # elif args.deep_ft_prompt:
        #     print("Using PromptedTransformer with deep feature prompt")
        #     model = PromptedTransformer(vit_kwargs, n_token, 0.,
        #                                 deep_prompt=False, deep_feature_prompt=True,
        #                                 project_prompt_dim=-1, prompt_aggregation=args.prompt_aggregation)
        # else:
        #     print("Using PromptedTransformer without deep prompt or deep feature prompt")

        # print(model.fc_norm, model.head)

    else:
        raise NotImplementedError

    # if args.replace_bn:
    #     if 'ResNet50' in model_name:
    #         model_from = torchvision.models.resnet50(pretrained=False)
    #         pretrained_sup_ckp = torch.load('results/IDH/ResNet50/resnet50_supervised.pth')
    #         from collections import OrderedDict
    #         pretrained_state_dict = OrderedDict()
    #         for k, v in pretrained_sup_ckp.items():
    #             name_list = k.split('.')
    #             if 'module' in k:
    #                 name = '.'.join(name_list[1:])
    #             else:
    #                 name = k
    #             pretrained_state_dict[name] = v
    #         model_from.load_state_dict(pretrained_state_dict)
    #         model = replace_bn_layers(model, model_from)
    #     elif 'ViT' in model_name:
    #         model_from = PromptedTransformer(vit_kwargs, n_token, 0., deep_prompt=False, project_prompt_dim=-1)
    #         pretrained_sup_ckp = torch.load('results/IDH/ViT_S_16/ViT_supervised.pth')
    #         model_from.load_state_dict(pretrained_sup_ckp)
    #         model = replace_ln_layers(model, model_from)

    # if transfer_type == "prompt":
    #     for k, p in model.named_parameters():
    #         if "prompt" not in k:
    #             p.requires_grad = False
    # elif transfer_type == "cls":
    #     for k, p in model.named_parameters():
    #         if "cls_token" not in k:
    #             p.requires_grad = False
    # elif transfer_type == "cls+prompt":
    #     for k, p in model.named_parameters():
    #         if "prompt" not in k and "cls_token" not in k:
    #             p.requires_grad = False
    # elif transfer_type == "non_prompt":
    #     for k, p in model.named_parameters():
    #         if "prompt" in k:
    #             p.requires_grad = False
    # if transfer_type == "end2end":
    #     print("Enable all parameters update during training")
    if transfer_type == 'frozen':
        for k, p in model.named_parameters():
            p.requires_grad = False
        print("Frozen all feature extracor parameters during training")
    elif transfer_type == "end2end":
        print("Enable all parameters update during training")
    elif transfer_type == 'eval':
        pass
    # print number of tunable parameter
    if 'FPT' in model_name:
        num_params = sum(p.numel() for p in model[1].parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of tunable parameters: {num_params}")
    return model

if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    # x2 = torch.rand(1, 3, 384, 384)
    # model = get_extractor('ViT_S_16_prompt')
    # y = model(x, pre_logits=True)
    # print(y.size())
    #
    # model = get_extractor('ViT_S_16')
    # y = model(x, pre_logits=True)
    # print(y.size())

    model = get_extractor('ResNet50_prompt')
    y = model(x)
    print(y.size())
    # num_classes_pretrained = getattr(model, 'num_classes', vit_kwargs.get('num_classes', 1000))
    # load_custom_pretrained(model, pretrained_cfg=ckp)
   #  "fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"
   # "head.mlp.0.weight", "head.mlp.0.bias", "head.mlp.2.weight", "head.mlp.2.bias", "head.mlp.4.weight", "head.mlp.4.bias", "head.last_layer.weight_g", "head.last_layer.weight_v"