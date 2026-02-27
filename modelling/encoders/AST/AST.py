import os

import torch
from torch import nn
from einops import repeat
from timm.models.vision_transformer import Block

import timm
from torch.nn.init import trunc_normal_

from data.kinetics.kinetics_dataloader import kinetics_dataloader
from utils.ast_utils import PatchEmbed, get_vit_dict
from utils.essential import get_cfg


class ASTModel(nn.Module):

    ARCH_CONFIGS = {
        'base': dict(embed_dims=768, depth=12, num_heads=12, mlp_ratio=4.0),
        'large': dict(embed_dims=1024, depth=24, num_heads=16, mlp_ratio=4.0),
    }

    def __init__(self,
                 pretrained_model_path,
                 embed_dims,
                 fstride,
                 tstride,
                 input_fdim,
                 input_tdim,
                 imagenet_pretrain,
                 model_size,
                 pretrain,
                 dropout_p,
                 depth,
                 num_heads,
                 mlp_ratio,
                 return_cls_token,
                 audio_pretrain,
                 arch='base',
                 weight_sharing='independent'):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # -------- architecture override --------
        assert arch in self.ARCH_CONFIGS, f"Unsupported arch '{arch}'. Choose from {list(self.ARCH_CONFIGS.keys())}"
        arch_cfg = self.ARCH_CONFIGS[arch]
        embed_dims = arch_cfg['embed_dims']
        depth = arch_cfg['depth']
        num_heads = arch_cfg['num_heads']
        mlp_ratio = arch_cfg['mlp_ratio']

        self.arch = arch
        self.weight_sharing = weight_sharing
        self.pretrained_model_path = pretrained_model_path
        self.embed_dims = embed_dims
        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.imagenet_pretrain = imagenet_pretrain
        self.model_size = model_size
        self.pretrain = pretrain
        self.dropout_p = dropout_p
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.return_cls_token = return_cls_token
        self.audio_pretrain = audio_pretrain

        self.norm_layer = nn.LayerNorm
        self.act_layer = nn.GELU

        # -------- patch_embed --------
        self.patch_embed = PatchEmbed(patch_size=(self.fstride, self.tstride), embed_dim=self.embed_dims)
        self.f_dim, self.t_dim = self.get_shape()
        num_patches = self.f_dim * self.t_dim
        self.num_patches = num_patches

        # -------- learnable pos_embed --------
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dims))
        self.drop_after_pos = nn.Dropout(p=self.dropout_p)

        # -------- cls_token & dist_token --------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # -------- Transformer Blocks --------
        self.blocks = nn.ModuleList([
            Block(dim=self.embed_dims,
                  num_heads=self.num_heads,
                  mlp_ratio=self.mlp_ratio,
                  qkv_bias=True,
                  qk_scale=None,
                  norm_layer=self.norm_layer,
                  act_layer=self.act_layer)
            for _ in range(self.depth)])

        self.norm = self.norm_layer(self.embed_dims, eps=1e-6)

        self.init_weights()

    def init_weights(self):
        # Initialize all parameters first
        self.apply(self._init_weights)

        trunc_normal_(self.patch_embed.proj.weight, std=.02)
        trunc_normal_(self.patch_embed.proj.bias, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if self.pretrain:
            old_state_dict={}
            if self.audio_pretrain:
                old_state_dict = torch.load(self.pretrained_model_path)
            elif not self.pretrain:
                if self.model_size == 'tiny224':
                    pretrained_model = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=self.imagenet_pretrain)
                elif self.model_size == 'small224':
                    pretrained_model = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=self.imagenet_pretrain)
                elif self.model_size == 'base224':
                    pretrained_model = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=self.imagenet_pretrain)
                elif self.model_size == 'base384':
                    pretrained_model = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=self.imagenet_pretrain)
                else:
                    raise Exception('Model size must be one of tiny224, small224, base224, base384.')
                old_state_dict = pretrained_model.state_dict()

            new_state_dict = get_vit_dict(old_state_dict=old_state_dict, embed_dims=self.embed_dims)

            # Filter out size-mismatched keys (e.g. base pretrained → large model)
            model_state = self.state_dict()
            compatible_dict = {k: v for k, v in new_state_dict.items()
                               if k in model_state and v.shape == model_state[k].shape}
            missing, unexpected = self.load_state_dict(compatible_dict, strict=False)
            if missing:
                print(f"[AST-{self.arch}] init_weights: {len(missing)} keys randomly initialized (not in pretrained).")

        else:
            print("Pretraining not implemented yet")

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_shape(self):
        test_input = torch.randn(1, 1, self.input_fdim, self.input_tdim)
        test_proj = nn.Conv2d(1, self.embed_dims, kernel_size=(16, 16), stride=(self.fstride, self.tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        x = x.float()

        x = x.unsqueeze(1)

        # -------- patch_embed --------
        x = self.patch_embed(x)

        # -------- learnable pos_embed --------
        x.add_(self.pos_embed)
        x = self.drop_after_pos(x)

        # -------- cls_token & dist_token --------
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        dist_token = self.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        # -------- Transformer Encoder --------
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if self.return_cls_token:
            return x[:, 0]
        else:
            return x[:, 1:].mean(1)
