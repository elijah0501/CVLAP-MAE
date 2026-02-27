import os

import torch
import torch.nn as nn
from einops import repeat, rearrange, reduce
from torch.nn.init import trunc_normal_
from transformers import AutoModelForPreTraining, AutoModelForImageClassification

from data.kinetics.kinetics_dataloader import kinetics_dataloader
from modelling.encoders.ViViT.transformer import BasicTransformerBlock
from utils.essential import get_cfg
from utils.vivit_utils import PatchEmbed, get_vit_dict, get_clip_dict


class ViViT(nn.Module):

    ARCH_CONFIGS = {
        'base': dict(embed_dims=768, num_heads=12, spatial_depth=12, temporal_depth=4),
        'large': dict(embed_dims=1024, num_heads=16, spatial_depth=24, temporal_depth=4),
    }

    def __init__(self,
                 input_size,
                 patch_size,
                 tube_size,
                 in_channels,
                 embed_dims,
                 spatial_depth,
                 temporal_depth,
                 num_heads,
                 num_frames,
                 dropout_p,
                 return_cls_token,
                 pretrained,
                 num_layers,
                 mae_pretrain,
                 vit_pretrain,
                 clip_pretrain,
                 arch='base',
                 weight_sharing='independent'
                 ):
        super(ViViT, self).__init__()

        # -------- architecture override --------
        assert arch in self.ARCH_CONFIGS, f"Unsupported arch '{arch}'. Choose from {list(self.ARCH_CONFIGS.keys())}"
        arch_cfg = self.ARCH_CONFIGS[arch]
        embed_dims = arch_cfg['embed_dims']
        num_heads = arch_cfg['num_heads']
        spatial_depth = arch_cfg['spatial_depth']
        temporal_depth = arch_cfg['temporal_depth']

        self.arch = arch
        self.weight_sharing = weight_sharing
        self.input_size = input_size
        self.patch_size = patch_size
        self.tube_size = tube_size
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.spatial_depth = spatial_depth
        self.temporal_depth = temporal_depth
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.dropout_p = dropout_p
        self.return_cls_token = return_cls_token
        self.pretrained = pretrained
        self.num_layers = num_layers
        self.mae_pretrain = mae_pretrain
        self.vit_pretrain = vit_pretrain
        self.clip_pretrain = clip_pretrain

        self.num_time_transformer_layers = 4

        self.norm_layer = nn.LayerNorm
        self.act_layer = nn.GELU

        # -------- patch_embed --------
        self.patch_embed = PatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dims=self.embed_dims,
            tube_size=self.tube_size)
        self.num_patches = self.patch_embed.num_patches

        # -------- learnable pos_embed --------
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dims))
        self.drop_after_pos = nn.Dropout(p=self.dropout_p)

        # -------- learnable time_embed --------
        self.time_embed = nn.Parameter(torch.zeros(1, self.num_frames // self.tube_size + 1, self.embed_dims))
        self.drop_after_time = nn.Dropout(p=self.dropout_p)

        # -------- cls_token --------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # -------- Transformer Encoder --------
        self.spatial_encoder = nn.ModuleList([
            BasicTransformerBlock(embed_dims=self.embed_dims,
                                  num_heads=self.num_heads,
                                  num_frames=self.num_frames,
                                  hidden_channels=self.embed_dims * 4,
                                  operator_order=['self_attn', 'ffn'],
                                  norm_layer=self.norm_layer,
                                  act_layer=self.act_layer,
                                  num_layers=self.num_layers,
                                  dpr=self.dropout_p)
            for _ in range(self.spatial_depth)])

        self.temporal_encoder = nn.ModuleList([
            BasicTransformerBlock(embed_dims=self.embed_dims,
                                  num_heads=self.num_heads,
                                  num_frames=self.num_frames,
                                  hidden_channels=self.embed_dims * 4,
                                  operator_order=['self_attn', 'ffn'],
                                  norm_layer=self.norm_layer,
                                  act_layer=self.act_layer,
                                  num_layers=self.num_layers,
                                  dpr=self.dropout_p)
            for _ in range(self.temporal_depth)])

        self.norm = self.norm_layer(self.embed_dims, eps=1e-6)

        self.init_weights()

    def init_weights(self):
        # Initialize all parameters first
        self.apply(self._init_weights)

        trunc_normal_(self.patch_embed.proj.weight, std=.02)
        trunc_normal_(self.patch_embed.proj.bias, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if self.mae_pretrain:
            model_name = "facebook/vit-mae-base" if self.arch == 'base' else "facebook/vit-mae-large"
            model = AutoModelForPreTraining.from_pretrained(model_name)
            old_state_dict = model.state_dict()

            new_dict = get_vit_dict(old_state_dict=old_state_dict,
                                    tube_size=self.tube_size,
                                    embed_dims=self.embed_dims,
                                    num_frames=self.num_frames)

            # Load state dict (strict=False: extra spatial layers beyond pretrained depth keep xavier init)
            missing, unexpected = self.load_state_dict(new_dict, strict=False)
            if missing:
                print(f"[ViViT-{self.arch}] init_weights: {len(missing)} keys randomly initialized (not in pretrained).")

        elif self.vit_pretrain:
            model_name = "google/vit-base-patch16-224" if self.arch == 'base' else "google/vit-large-patch16-224"
            model = AutoModelForImageClassification.from_pretrained(model_name)
            old_state_dict = model.state_dict()

            new_dict = get_vit_dict(old_state_dict=old_state_dict,
                                    tube_size=self.tube_size,
                                    embed_dims=self.embed_dims,
                                    num_frames=self.num_frames)

            # Load state dict (strict=False: extra spatial layers beyond pretrained depth keep xavier init)
            missing, unexpected = self.load_state_dict(new_dict, strict=False)
            if missing:
                print(f"[ViViT-{self.arch}] init_weights: {len(missing)} keys randomly initialized (not in pretrained).")

        elif self.clip_pretrain:
            model_name = "openai/clip-vit-base-patch32" if self.arch == 'base' else "openai/clip-vit-large-patch14"
            model = AutoModelForImageClassification.from_pretrained(model_name)
            old_state_dict = model.state_dict()

            new_dict = get_clip_dict(old_state_dict=old_state_dict,
                                     tube_size=self.tube_size,
                                     embed_dims=self.embed_dims,
                                     num_frames=self.num_frames,
                                     num_patches=self.num_patches,
                                     in_channels=self.in_channels,
                                     patch_size=self.patch_size)

            # Load state dict (strict=False: extra spatial layers beyond pretrained depth keep xavier init)
            missing, unexpected = self.load_state_dict(new_dict, strict=False)
            if missing:
                print(f"[ViViT-{self.arch}] init_weights: {len(missing)} keys randomly initialized (not in pretrained).")

        else:
            print("Pretraining not implemented yet")

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1, 3, 4)
        b = x.shape[0]

        # -------- patch_embed --------
        x = self.patch_embed(x)

        # -------- cls_tokens --------
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # -------- pos_embed --------
        x.add_(self.pos_embed)

        x = self.drop_after_pos(x)

        #  -------- spatial transformer --------
        for block in self.spatial_encoder:
            x = block(x)
        x = self.norm(x)

        #  -------- time_embed --------
        # Convert the output of the spatial dimension to the input on the time dimension and add cls_token
        cls_tokens = x[:b, 0, :].unsqueeze(1)
        x = rearrange(x[:, 1:, :], '(b t) p d -> b t p d', b=b)
        x = x.mean(dim=2)
        x = torch.cat((cls_tokens, x), dim=1)

        x.add_(self.time_embed)
        x = self.drop_after_time(x)

        #  -------- temporal transformer --------
        for block in self.temporal_encoder:
            x = block(x)
        x = self.norm(x)

        if self.return_cls_token:
            return x[:, 0]
        else:
            return x[:, 1:].mean(1)
