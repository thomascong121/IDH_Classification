#!/usr/bin/env python3

"""
ResNet-related models:
"imagenet_sup_rn18",
"imagenet_sup_rn34",
"imagenet_sup_rn50",
"imagenet_sup_rn101",
"imagenet_sup_rn152",
"mocov3_rn50"
"""
import sys
sys.path.insert(1, '/scratch/sz65/cc0395/WSI_prompt/')
import torch
import torch.nn as nn
import torchvision as tv
from functools import reduce
from itertools import repeat
from operator import mul
import math, collections
from collections import OrderedDict
from torchvision import models
from model.FExtractor.mlp import MLP


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")
class PromptResNet(nn.Module):
    """ResNet model."""

    def __init__(self, cfg, pretrained_model=None):
        super(PromptResNet, self).__init__()
        self.cfg = cfg

        model_type = cfg.ft_model
        if pretrained_model is not None:
            model = pretrained_model
        else:
            model = self.get_pretrained_model(model_type)
        named_layers = dict(model.named_modules())
        if 'layer4' not in named_layers:
            model.layer4 = nn.Identity()
        last_conv_layer = model.layer3[-1].conv3
        self.emb_length = last_conv_layer.weight.size(0)
        if "prompt" in cfg.transfer_type:
            # setup prompt_embd and modify model accordingly
            model = self.setup_prompt(cfg, model)
            self.prompt_aggregation = cfg.prompt_aggregation
            self.prompt_location = cfg.prompt_location
        else:
            self.prompt_embeddings = None

        # setup side network if needed
        self.setup_side()
        # set which parameters require grad
        # creat self.prompt_layers, self.frozen_layers, self.tuned_layers
        self.setup_grad(model)

    def setup_side(self):
        if self.cfg.transfer_type != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            out_dim = self.get_outputdim()
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, out_dim, bias=False)

    def setup_grad(self, model):
        transfer_type = self.cfg.transfer_type
        # split enc into 3 parts:
        #           prompt_layers  frozen_layers  tuned_layers
        # partial-1  identity       -layer3       layer4
        # partial-2: identity       -layer2      "layer4" "layer3"
        # partial-3: identity       -layer1      "layer4" "layer3" "layer2"
        # linear     identity        all          identity
        # end2end    identity       identity      all

        # prompt-below  conv1        all but conv1
        # prompt-pad   identity        all
        if transfer_type == "prompt" and self.prompt_location == "below": # noqa
            self.prompt_layers = nn.Sequential(OrderedDict([
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("relu", model.relu),
                ("maxpool", model.maxpool),
            ]))
            self.frozen_layers = nn.Sequential(OrderedDict([
                ("layer1", model.layer1),
                ("layer2", model.layer2),
                ("layer3", model.layer3),
                ("layer4", model.layer4),
                ("avgpool", model.avgpool),
            ]))
            self.tuned_layers = nn.Identity()
        else:
            # partial, linear, end2end, prompt-pad
            self.prompt_layers = nn.Identity()

            if transfer_type == "partial-0":
                # last conv block
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4[:-1]),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer4", model.layer4[-1]),
                    ("avgpool", model.avgpool),
                ]))
            elif transfer_type == "partial-1":
                # tune last layer
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            elif transfer_type == "partial-2":
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            elif transfer_type == "partial-3":
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                ]))
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            # elif transfer_type == "linear" or transfer_type == "side" or transfer_type == "tinytl-bias":
            elif transfer_type in ['linear', 'side', 'tinytl-bias', 'eval']:
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))
                self.tuned_layers = nn.Identity()

            elif transfer_type == "end2end":
                self.frozen_layers = nn.Identity()
                self.tuned_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))

            elif transfer_type == "prompt" and self.prompt_location == "pad": # noqa
                print('====> Setup prompt-pad')
                self.frozen_layers = nn.Sequential(OrderedDict([
                    ("conv1", model.conv1),
                    ("bn1", model.bn1),
                    ("relu", model.relu),
                    ("maxpool", model.maxpool),
                    ("layer1", model.layer1),
                    ("layer2", model.layer2),
                    ("layer3", model.layer3),
                    ("layer4", model.layer4),
                    ("avgpool", model.avgpool),
                ]))
                self.tuned_layers = nn.Identity()
            elif transfer_type == "prompt" and self.prompt_location == "deep":  # noqa
                self.frozen_layers = nn.Sequential(OrderedDict([
                        ("conv1", model.conv1),
                        ("bn1", model.bn1),
                        ("relu", model.relu),
                        ("maxpool", model.maxpool),
                        ("layer1", model.layer1),
                        ("layer2", model.layer2),
                        ("layer3", model.layer3),
                        ("layer4", model.layer4),
                        ("avgpool", model.avgpool),
                    ]))
                self.tuned_layers = nn.Identity()
        if transfer_type == "tinytl-bias":
            print('====> Setup tinytl-bias')
            for k, p in self.frozen_layers.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False
        else:
            for k, p in self.frozen_layers.named_parameters():
                p.requires_grad = False
        self.transfer_type = transfer_type

    def setup_prompt(self, prompt_config, model):
        self.prompt_location = prompt_config.prompt_location
        self.prompt_location = 'deep' if prompt_config.deep_ft_prompt else self.prompt_location
        self.num_tokens = prompt_config.number_prompts
        print('====> Setup %d %s prompt with %s init'%(self.num_tokens, self.prompt_location, prompt_config.prompt_initialisation))
        if self.prompt_location == "below":
            return self._setup_prompt_below(prompt_config, model)
        elif self.prompt_location == "pad":
            return self._setup_prompt_pad(prompt_config, model)
        elif self.prompt_location == "deep":
            return self._setup_prompt_deep(prompt_config, model)
        else:
            raise ValueError(
                "ResNet models cannot use prompt location {}".format(
                    prompt_config.prompt_location))

    def _setup_prompt_below(self, prompt_config, model):
        if prompt_config.prompt_initialisation == "random":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.cfg.image_size, self.cfg.image_size
            ))
            nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
            self.prompt_norm = tv.transforms.Normalize(
                mean=[sum([0.485, 0.456, 0.406])/3] * self.num_tokens,# self.num_tokens
                std=[sum([0.229, 0.224, 0.225])/3] * self.num_tokens,
            )

        elif prompt_config.prompt_initialisation == "gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens,
                    self.cfg.image_size, self.cfg.image_size
            ))

            nn.init.normal_(self.prompt_embeddings.data)

            self.prompt_norm = nn.Identity()

        else:
            raise ValueError("Other initiation scheme is not supported")

        # modify first conv layer
        old_weight = model.conv1.weight  # [64, 3, 7, 7]
        model.conv1 = nn.Conv2d(
            self.num_tokens+3, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        torch.nn.init.xavier_uniform(model.conv1.weight)

        model.conv1.weight[:, :3, :, :].data.copy_(old_weight)
        return model

    def _setup_prompt_pad(self, prompt_config, model):
        if prompt_config.prompt_initialisation == "random":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.num_tokens,
                    self.cfg.image_size + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, self.cfg.image_size, 2 * self.num_tokens
            ))

            nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
            nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

            self.prompt_norm = tv.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        elif prompt_config.prompt_initialisation == "gaussian":
            self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * self.num_tokens,
                    self.cfg.image_size + 2 * self.num_tokens
            ))
            self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, self.cfg.image_size, 2 * self.num_tokens
            ))

            nn.init.normal_(self.prompt_embeddings_tb.data)
            nn.init.normal_(self.prompt_embeddings_lr.data)

            self.prompt_norm = nn.Identity()
        else:
            raise ValueError("Other initiation scheme is not supported")
        return model

    def _setup_prompt_deep(self, prompt_config, model):
        patch_size = _pair(16)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.emb_length))  # noqa
        if prompt_config.prompt_initialisation == "random":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens, self.emb_length
            ))
            nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
            self.prompt_norm = tv.transforms.Normalize(
                mean=[sum([0.485, 0.456, 0.406])] * 1,# /3, self.num_tokens
                std=[sum([0.229, 0.224, 0.225])] * 1,
            )

        elif prompt_config.prompt_initialisation == "gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, self.num_tokens, self.emb_length
            ))
            # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            nn.init.normal_(self.prompt_embeddings.data)
            self.prompt_norm = nn.Identity()
        else:
            raise ValueError("Other initiation scheme is not supported")
        return model


    def get_pretrained_model(self, model_type):
        model_root = self.cfg.MODEL.MODEL_ROOT

        if model_type == "imagenet_sup_rn50":
            model = models.resnet50(pretrained=True)
        elif model_type == "imagenet_sup_rn101":
            model = models.resnet101(pretrained=True)  # 2048
        elif model_type == "imagenet_sup_rn152":
            model = models.resnet152(pretrained=True)  # 2048
        elif model_type == "imagenet_sup_rn34":
            model = models.resnet34(pretrained=True)   # 512
        elif model_type == "imagenet_sup_rn18":
            model = models.resnet18(pretrained=True)   # 512

        elif model_type == "inat2021_sup_rn50":
            checkpoint = torch.load(
                f"{model_root}/inat2021_supervised_large.pth.tar",
                map_location=torch.device('cpu')
            )
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 10000)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        elif model_type == 'inat2021_mini_sup_rn50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 10000)
            checkpoint = torch.load(
                f"{model_root}/inat2021_supervised_mini.pth.tar",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(checkpoint['state_dict'], strict=True)

        elif model_type == 'inat2021_mini_moco_v2_rn50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Identity()
            checkpoint = torch.load(
                f"{model_root}/inat2021_moco_v2_mini_1000_ep.pth.tar",
                map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict, strict=True)

        elif model_type == 'imagenet_moco_v2_rn50':
            model = models.resnet50(pretrained=False)
            model.fc = torch.nn.Identity()
            checkpoint = torch.load(
                f"{model_root}/imagenet_moco_v2_800ep_pretrain.pth.tar",
                map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            model.load_state_dict(state_dict, strict=True)

        elif model_type.startswith("mocov3_rn50"):
            moco_epoch = model_type.split("ep")[-1]
            checkpoint = torch.load(
                f"{model_root}/mocov3_linear-1000ep.pth.tar",
                map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model = models.resnet50()
            model.load_state_dict(state_dict, strict=False)

        else:
            raise ValueError("model type not supported for resnet backbone")

        model.fc = nn.Identity()
        return model

    def get_outputdim(self):
        # if self.cfg.DATA.FEATURE == "imagenet_sup_rn34" or self.cfg.DATA.FEATURE == "imagenet_sup_rn18":
        #     out_dim = 512
        # else:
        #     out_dim = 2048
        out_dim = 1024
        return out_dim

    def setup_head(self, cfg):
        out_dim = self.get_outputdim()
        self.head = MLP(
            input_dim=out_dim,
            mlp_dims=[out_dim] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES],
            special_bias=True
        )

    def incorporate_prompt(self, x):
        B = x.shape[0]
        if self.prompt_location == "below":
            x = torch.cat((
                    x,
                    self.prompt_norm(
                        self.prompt_embeddings).expand(B, -1, -1, -1),
                ), dim=1)
            # (B, 3 + num_prompts, crop_size, crop_size)

        elif self.prompt_location == "pad":
            prompt_emb_lr = self.prompt_norm(
                self.prompt_embeddings_lr).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self.prompt_embeddings_tb).expand(B, -1, -1, -1)

            x = torch.cat((
                prompt_emb_lr[:, :, :, :self.num_tokens],
                x, prompt_emb_lr[:, :, :, self.num_tokens:]
                ), dim=-1)
            x = torch.cat((
                prompt_emb_tb[:, :, :self.num_tokens, :],
                x, prompt_emb_tb[:, :, self.num_tokens:, :]
            ), dim=-2)
            # (B, 3, crop_size + num_prompts, crop_size + num_prompts)
        elif self.prompt_location == 'deep':
            return x
        else:
            raise ValueError("not supported yet")
        x = self.prompt_layers(x)
        return x

    def forward(self, x, return_feature=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        x = self.get_features(x)

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        if return_feature:
            return x

        return self.head(x)

    def get_features(self, x):
        """get a (batch_size, 2048) feature"""
        if self.frozen_layers.training:
            self.frozen_layers.eval()

        if "prompt" not in self.transfer_type:
            with torch.set_grad_enabled(self.frozen_layers.training):
                x = self.frozen_layers(x)
        else:
            # prompt tuning required frozen_layers saving grad
            x = self.incorporate_prompt(x)
            x = self.frozen_layers(x)
        x = self.tuned_layers(x)  # batch_size x 2048 x 1
        x = x.view(x.size(0), -1)
        if "prompt" in self.transfer_type and self.prompt_location == 'deep':
            B = x.size(0)
            prompt = self.prompt_norm(
                        self.prompt_embeddings).expand(B, -1, -1)
            prompt = torch.permute(prompt, (1, 0, 2))
            x = x.expand(prompt.size(0), -1, -1)
            if self.prompt_aggregation == "add":
                x = x + prompt
            elif self.prompt_aggregation == "prepend":
                x = torch.cat((prompt, x), dim=1)
            elif self.prompt_aggregation == "multiply":
                x = x * prompt
            else:
                raise NotImplementedError
            x = torch.mean(x, dim=0)

        return x

if __name__ == '__main__':
    m = models.alexnet(pretrained=True)