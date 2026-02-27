import os
from typing import Tuple

import einops
import timm
import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_
from timm.models.vision_transformer import Block

from data.kinetics.kinetics_dataloader import kinetics_dataloader
from utils.ast_utils import PatchEmbed, get_vit_dict_mae
from utils.essential import get_cfg


class ASTMAE(nn.Module):
    """Audio Spectrogram Transformer Masked Autoencoder.

    Implements MAE pre-training for audio spectrograms using a ViT-based
    encoder-decoder architecture with random patch masking.
    """

    def __init__(self,
                 fstride: int,
                 tstride: int,
                 encoder_embed_dim: int,
                 dropout_p: float,
                 encoder_num_head: int,
                 mlp_ratio: float,
                 encoder_depth: int,
                 decoder_embed_dim: int,
                 decoder_num_head: int,
                 decoder_depth: int,
                 pretrain: bool,
                 audio_pretrain: bool,
                 imagenet_pretrain: bool,
                 pretrained_model_path: str,
                 input_fdim: int,
                 input_tdim: int,
                 mask_ratio: float,
                 norm_pix_loss: bool,
                 model_size: str = 'base384',
                 arch: str = 'base',
                 **kwargs):
        super().__init__()
        assert timm.__version__ == '0.4.5', \
            'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # Override parameters if arch is 'large'
        if arch == 'large':
            encoder_embed_dim = 1024
            encoder_num_head = 16
            encoder_depth = 24

        self.arch = arch
        self.fstride = fstride
        self.tstride = tstride
        self.encoder_embed_dim = encoder_embed_dim
        self.dropout_p = dropout_p
        self.encoder_num_head = encoder_num_head
        self.mlp_ratio = mlp_ratio
        self.encoder_depth = encoder_depth

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_head = decoder_num_head
        self.decoder_depth = decoder_depth

        self.pretrain = pretrain
        self.audio_pretrain = audio_pretrain
        self.imagenet_pretrain = imagenet_pretrain
        self.pretrained_model_path = pretrained_model_path
        self.model_size = model_size
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        # Compute patch grid dimensions via arithmetic (avoids temporary Conv2d + tensor allocation)
        self.patch_size = (self.fstride, self.tstride)
        self.f_dim = self.input_fdim // self.fstride
        self.t_dim = self.input_tdim // self.tstride
        self.num_patches = self.f_dim * self.t_dim

        # -----------------------------------------------------------------
        # AST_MAE Encoder

        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size, embed_dim=self.encoder_embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.encoder_embed_dim))
        self.drop_after_pos = nn.Dropout(p=self.dropout_p)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(dim=self.encoder_embed_dim,
                  num_heads=self.encoder_num_head,
                  mlp_ratio=self.mlp_ratio,
                  qkv_bias=True,
                  qk_scale=None,
                  norm_layer=nn.LayerNorm,
                  act_layer=nn.GELU)
            for _ in range(self.encoder_depth)])

        self.encoder_norm = nn.LayerNorm(self.encoder_embed_dim, eps=1e-6)

        # -----------------------------------------------------------------
        # AST_MAE Decoder

        self.decoder_embed = nn.Linear(
            self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim),
            requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(dim=self.decoder_embed_dim,
                  num_heads=self.decoder_num_head,
                  mlp_ratio=self.mlp_ratio,
                  qkv_bias=True,
                  qk_scale=None,
                  norm_layer=nn.LayerNorm,
                  act_layer=nn.GELU)
            for _ in range(self.decoder_depth)])

        self.decoder_norm = nn.LayerNorm(self.decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(
            self.decoder_embed_dim,
            self.patch_embed.patch_size[0] ** 2 * 1,
            bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self._initialize_weights()

    @torch.no_grad()
    def _initialize_weights(self):
        """Initialize model parameters, optionally loading pretrained weights."""
        self.apply(self._init_weights)

        trunc_normal_(self.patch_embed.proj.weight, std=.02)
        trunc_normal_(self.patch_embed.proj.bias, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if self.pretrain:
            if self.audio_pretrain:
                old_state_dict = torch.load(
                    self.pretrained_model_path, map_location='cpu')
            elif self.imagenet_pretrain:
                # FIX: original code used `elif not self.pretrain` which was
                # unreachable inside `if self.pretrain`; corrected to check
                # `self.imagenet_pretrain` as design intended.
                _model_name_map = {
                    'tiny224': 'vit_deit_tiny_distilled_patch16_224',
                    'small224': 'vit_deit_small_distilled_patch16_224',
                    'base224': 'vit_deit_base_distilled_patch16_224',
                    'base384': 'vit_deit_base_distilled_patch16_384',
                }
                if self.model_size not in _model_name_map:
                    raise ValueError(
                        f'Model size must be one of {list(_model_name_map.keys())}, '
                        f'got {self.model_size!r}.')
                pretrained_model = timm.create_model(
                    _model_name_map[self.model_size], pretrained=True)
                old_state_dict = pretrained_model.state_dict()
            else:
                print("No pretrained weights specified, using random initialization.")
                return

            new_state_dict = get_vit_dict_mae(
                old_state_dict=old_state_dict,
                embed_dims=self.encoder_embed_dim,
                num_patches=self.num_patches,
                decoder_embed_dim=self.decoder_embed_dim,
                pred_dims=self.patch_embed.patch_size[0] ** 2 * 1)

            # Filter out size-mismatched keys (e.g. base pretrained → large model)
            model_state = self.state_dict()
            compatible_dict = {k: v for k, v in new_state_dict.items()
                               if k in model_state and v.shape == model_state[k].shape}
            missing, unexpected = self.load_state_dict(compatible_dict, strict=False)
            if missing:
                print(f"[ASTMAE] init_weights: {len(missing)} keys randomly initialized (not in pretrained).")
        else:
            print("Pretraining not implemented yet")

    @staticmethod
    def _init_weights(m: nn.Module):
        """Xavier uniform init for Linear layers, constant init for LayerNorm."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform per-sample random masking by per-sample shuffling.

        Args:
            x: Input tensor of shape [N, L, D].

        Returns:
            x_masked: Masked input [N, len_keep, D].
            mask: Binary mask [N, L], 0 is keep, 1 is removed.
            ids_restore: Indices to restore original ordering [N, L].
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        # Use expand instead of repeat for index tensor (avoids memory copy)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs: Tensor) -> Tensor:
        """Convert spectrogram images to patch sequences.

        Args:
            imgs: Input of shape (N, H, W).

        Returns:
            Patches of shape (N, num_patches, patch_size**2).
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[1] % p == 0 and imgs.shape[2] % p == 0

        # Use einops.rearrange for clarity instead of manual reshape + einsum
        h = imgs.shape[1] // p
        w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 1, h, p, w, p)
        x = einops.rearrange(x, 'n c h p1 w p2 -> n (h w) (p1 p2 c)')
        return x

    def forward_encoder(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = x.float()
        x = x.unsqueeze(1).transpose(2, 3)

        x = self.patch_embed(x)
        after_embed = x

        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # masking: length -> length * (1 - mask_ratio)
        x, mask, ids_restore = self.random_masking(x)

        # Prepend cls_token and dist_token; expand creates a view (no memory copy)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        dist_tokens = self.dist_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore, after_embed

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.decoder_embed(x)

        # Append mask tokens to sequence (skip cls & dist tokens at positions 0, 1)
        num_mask_tokens = ids_restore.shape[1] + 2 - x.shape[1]
        mask_tokens = self.mask_token.expand(x.shape[0], num_mask_tokens, -1)
        x_ = torch.cat([x[:, 2:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token

        return x, x_

    def forward_loss(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """Compute MSE reconstruction loss on masked patches.

        Args:
            imgs: Original spectrogram [N, H, W].
            pred: Predicted patches [N, L, p*p].
            mask: Binary mask [N, L], 0=keep, 1=removed.

        Returns:
            Mean reconstruction loss on masked patches.
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent, mask, ids_restore, after_embed = self.forward_encoder(x)
        pred, masked = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask
