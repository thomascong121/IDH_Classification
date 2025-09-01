from copy import deepcopy
from functools import reduce
from operator import mul
import math
from collections import OrderedDict
from torchvision import models
import timm.models.vision_transformer as vit
import torch
from .mlp import MLP
# from timm.models.helpers import update_pretrained_cfg_and_kwargs, load_pretrained, load_custom_pretrained
from utils.core_util import replace_bn_layers, replace_ln_layers, custom_load_pretrained, load_pretrained_vit
# from timm.models.vision_transformer import default_cfgs, checkpoint_filter_fn
from torch import nn
from torch.nn.modules.utils import _pair
from model.FExtractor.vit_adp import ADPTTransformer
from model.FExtractor.vit_base import BaseTransformer

class PromptedTransformer(vit.VisionTransformer):
    """ Vision Transformer

        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        """

    def __init__(
            self,
            vit_config,
            num_tokens=1,
            drop_out=0.,
            project_prompt_dim=-1,
            deep_prompt=False,
            deep_feature_prompt=False,
            prompt_aggregation="multiply",
            # freeze_backbone=True
    ):
        super().__init__(**vit_config)
        # if freeze_backbone:
        #     for param in self.parameters():
        #         param.requires_grad = False
        # self.prompt_config = prompt_config
        self.vit_config = vit_config
        self.prompt_aggregation = prompt_aggregation
        patch_size = _pair(vit_config["patch_size"])

        self.num_prompt_tokens = num_tokens  # number of prompted tokens
        self.deep_prompt = deep_prompt
        self.deep_feature_prompt = deep_feature_prompt
        self.prompt_dropout = nn.Dropout(drop_out)

        # if project the prompt embeddings
        if project_prompt_dim > 0:
            # only for prepend / add
            prompt_dim = project_prompt_dim
            self.prompt_proj = nn.Linear(
                prompt_dim, vit_config["embed_dim"])
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim = vit_config["embed_dim"]
            self.prompt_proj = nn.Identity()
        if num_tokens > 0:
            # initiate input prompt:
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            if not self.deep_feature_prompt:
                self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        else:
            pass

        # if self.deep_prompt:  # noqa
        #     total_d_layer = vit_config["depth"] - 1
        #     self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
        #         total_d_layer, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        #
        # if self.deep_feature_prompt:
        #     self.deep_ft_prompt_embeddings = nn.Parameter(torch.zeros(
        #         1, num_tokens, prompt_dim))
        #     # xavier_uniform initialization
        #     nn.init.uniform_(self.deep_ft_prompt_embeddings.data, -val, val)
        self.norm = nn.LayerNorm(vit_config["embed_dim"], eps=1e-6)
        self.head = nn.Identity()
        # saved_args = locals()
        # print("saved_args is", saved_args)
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = self.patch_embed(x)
        x = self._pos_embed(x)  # (batch_size, 1 + n_patches, hidden_dim)

        prompt = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1))
        # if self.num_prefix_tokens > 0:
        # num_prefix_tokens: number of class tokens
        x = torch.cat((
            x[:, :self.num_prefix_tokens, :],
            prompt,
            x[:, self.num_prefix_tokens:, :]
        ), dim=1)
        # else:
        #     x = torch.cat((prompt, x[:, :, :]), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        x = self.norm_pre(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            if self.num_prompt_tokens > 0:
                super().train(False)
                self.prompt_proj.train()
                self.prompt_dropout.train()
            else:
                super().train(mode)
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        # attn_weights = []
        hidden_states = None
        # weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config["depth"]

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.blocks[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :self.num_prefix_tokens, :],
                        deep_prompt_emb,
                        hidden_states[:, (self.num_prefix_tokens + self.num_prompt_tokens):, :]
                    ), dim=1)

                hidden_states = self.blocks[i](hidden_states)

            # if self.encoder.vis:
            #     attn_weights.append(weights)

        # encoded = self.encoder.encoder_norm(hidden_states)
        # return encoded, attn_weights
        return hidden_states

    def forward_features(self, x):
        if self.num_prompt_tokens > 0 and not self.deep_feature_prompt:
            x = self.incorporate_prompt(x)
        else:
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.norm_pre(x)

        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        #     x = self.blocks(x)
        if self.num_prompt_tokens > 0 and self.deep_prompt:
            # x, attn_weights = self.forward_deep_prompt(x)
            x = self.forward_deep_prompt(x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, return_feature=False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens + self.num_prompt_tokens:].mean(dim=1) \
                if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        # if self.deep_feature_prompt:
        #     B = x.size(0)
        #     prompt = self.prompt_dropout(self.prompt_proj(self.deep_ft_prompt_embeddings).expand(B, -1, -1))
        #     # prompt size (B, num_tokens, hidden_dim), x size (B, hidden_dim)
        #     prompt = torch.permute(prompt, (1,0,2))
        #     x = x.unsqueeze(0).expand(prompt.size(0), -1, -1)
        #     if self.prompt_aggregation == "add":
        #         x = x + prompt
        #     elif self.prompt_aggregation == "prepend":
        #         x = torch.cat((prompt, x), dim=1)
        #     elif self.prompt_aggregation == "multiply":
        #         x = x * prompt
        #     else:
        #         raise NotImplementedError
        #     x = torch.mean(x, dim=0)
        return x if return_feature else self.head(x)

    def forward(self, x, return_feature=False):
        x = self.forward_features(x)
        x = self.forward_head(x, return_feature)
        return x

class ViT(nn.Module):
    def __init__(self, cfg, vit_kwargs, variant, pretrained):
        super(ViT, self).__init__()
        if "prompt" in cfg.transfer_type:
            self.prompt_location = cfg.prompt_location
            model = PromptedTransformer(vit_kwargs, cfg.number_prompts, 0.,
                                        project_prompt_dim=-1)
            self.enc = load_pretrained_vit(model, cfg.ft_model, variant, vit_kwargs, cfg.custom_pretrained,
                                        pretrained=pretrained)
        elif "adapter" in cfg.transfer_type:
            adpt_kwargs = dict(STYLE="Pfeiffer", REDUCATION_FACTOR=cfg.adpt_rf)
            self.enc = ADPTTransformer(vit_kwargs, cfg.ft_model, variant, adpt_kwargs)
        else:
            model = BaseTransformer(vit_kwargs)
            self.enc = load_pretrained_vit(model, cfg.ft_model, variant, vit_kwargs, cfg.custom_pretrained,
                                        pretrained=pretrained)

        self.feat_dim = 384 if 'ViT_S_16' in cfg.ft_model else 192
        self.transfer_type = cfg.transfer_type
        if cfg.transfer_type != "end2end" and "prompt" not in cfg.transfer_type:
            # linear, cls, tiny-tl, parital, adapter
            self.froze_enc = True
        else:
            # prompt, end2end, cls+prompt
            self.froze_enc = False
        self.build_backbone()
        self.cfg = cfg
        self.setup_side()
        self.setup_head(cfg)

    def setup_side(self):
        if self.cfg.transfer_type != "side":
            self.side = None
        else:
            self.side_alpha = nn.Parameter(torch.tensor(0.0))
            m = models.alexnet(pretrained=True)
            self.side = nn.Sequential(OrderedDict([
                ("features", m.features),
                ("avgpool", m.avgpool),
            ]))
            self.side_projection = nn.Linear(9216, self.feat_dim, bias=False)

    def build_backbone(self):
        # linear, prompt, cls, cls+prompt, partial_1
        if self.transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer)
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False
        elif self.transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False

        elif self.transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 2) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 3) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False

        elif self.transfer_type == "linear" or self.transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif self.transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt" and self.prompt_location == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt-noupdate":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif self.transfer_type == "cls":
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "cls-reinit":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )

            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "cls+prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "cls-reinit+prompt":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        # adapter
        elif self.transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "end2end":
            print("Enable all parameters update during training")
        elif self.transfer_type == "eval":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False
        else:
            raise ValueError("transfer type {} is not supported".format(
                self.transfer_type))
    def setup_head(self, cfg):
        # self.head = MLP(
        #     input_dim=self.feat_dim,
        #     mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
        #         [cfg.DATA.NUMBER_CLASSES], # noqa
        #     special_bias=True
        # )
        self.head = nn.Identity()

    def forward(self, x, return_feature=False):
        if self.side is not None:
            side_output = self.side(x)
            side_output = side_output.view(side_output.size(0), -1)
            side_output = self.side_projection(side_output)

        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim

        if self.side is not None:
            alpha_squashed = torch.sigmoid(self.side_alpha)
            x = alpha_squashed * x + (1 - alpha_squashed) * side_output

        if return_feature:
            return x
        x = self.head(x)

        return x





if __name__ == '__main__':
    # model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, global_pool='avg',num_classes=2)

    pretrained = True

    variant = "vit_tiny_patch16_384"
    #
    model = PromptedTransformer(model_kwargs, 1, 0., deep_prompt=False, project_prompt_dim=-1)

    # def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    #     if not os.path.isfile(pretrained_weights):
    #         print("wrong weight path")
    #     else:
    #         state_dict = torch.load(pretrained_weights, map_location="cpu")
    #         if checkpoint_key is not None and checkpoint_key in state_dict:
    #             print(f"Take key {checkpoint_key} in provided checkpoint dict")
    #             state_dict = state_dict[checkpoint_key]
    #         # remove `module.` prefix
    #         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #         # remove `backbone.` prefix induced by multicrop wrapper
    #         state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    #         msg = model.load_state_dict(state_dict, strict=False)
    #         print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    #
    #
    # lung_dino_path = "/data04/shared/skapse/Cell_guided/Experiments/Lung_cancer/DINO_5X/100_percent_data_ep100/vit_tiny_baseline_avgpool_fp16true_momentum996_outdim65536/checkpoint.pth"
    #
    # load_pretrained_weights(model, lung_dino_path, "teacher")

    pretrained_cfg = deepcopy(default_cfgs[variant])

    update_pretrained_cfg_and_kwargs(pretrained_cfg, model_kwargs, None)
    pretrained_cfg.setdefault('architecture', variant)

    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    num_classes_pretrained = getattr(model, 'num_classes', model_kwargs.get('num_classes', 1000))

    pretrained_custom_load = 'npz' in pretrained_cfg['url']
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=model_kwargs.get('in_chans', 3),
                filter_fn=checkpoint_filter_fn,
                strict=False)

    transfer_type = "prompt"
    if transfer_type == "prompt":
        for k, p in model.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False
    elif transfer_type == "cls":
        for k, p in model.named_parameters():
            if "cls_token" not in k:
                p.requires_grad = False
    elif transfer_type == "cls+prompt":
        for k, p in model.named_parameters():
            if "prompt" not in k and "cls_token" not in k:
                p.requires_grad = False
    elif transfer_type == "end2end":
        print("Enable all parameters update during training")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name) #, p.data)

    x = torch.randn(1, 3, 224, 224)
    print(model(x, return_feature=True).shape)
