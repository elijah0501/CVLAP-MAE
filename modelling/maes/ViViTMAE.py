import os
from typing import Tuple

import torch
import torch.nn as nn
from einops import repeat, rearrange, reduce
from torch import Tensor
from torch.nn.init import trunc_normal_
from transformers import AutoModelForPreTraining, AutoModelForImageClassification

from data.kinetics.kinetics_dataloader import kinetics_dataloader
from modelling.encoders.ViViT.transformer import BasicTransformerBlock
from utils.essential import get_cfg
from utils.vivit_utils import PatchEmbed, get_vit_dict_mae, get_clip_dict_mae


class ViViTMAE(nn.Module):
    """Video Vision Transformer Masked Autoencoder.

    Implements MAE pre-training for video using a factorised spatial-temporal
    ViViT encoder and a spatial-only decoder, with random patch masking.
    """

    def __init__(self,
                 input_size: int,
                 patch_size: int,
                 tube_size: int,
                 in_channels: int,
                 embed_dims: int,
                 spatial_depth: int,
                 temporal_depth: int,
                 num_heads: int,
                 num_frames: int,
                 dropout_p: float,
                 return_cls_token: bool,
                 pretrained: bool,
                 num_layers: int,
                 mae_pretrain: bool,
                 vit_pretrain: bool,
                 clip_pretrain: bool,
                 mask_ratio: float,
                 arch: str = 'base',
                 norm_pix_loss: bool = False,
                 **kwargs):  # Accept extra config keys without error
        super().__init__()

        # Override parameters if arch is 'large'
        if arch == 'large':
            embed_dims = 1024
            num_heads = 16
            spatial_depth = 24
            # temporal_depth can remain 4 or be scaled depending on design

        self.arch = arch
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
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss

        norm_layer = nn.LayerNorm
        act_layer = nn.GELU

        # -------- patch_embed --------
        self.patch_embed = PatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dims=self.embed_dims,
            tube_size=self.tube_size)
        self.num_patches = self.patch_embed.num_patches

        # Pre-compute decoder spatial token count to eliminate magic numbers
        self._num_visible = int(self.num_patches * (1 - self.mask_ratio)) + 1
        self._num_spatial_with_cls = self.num_patches + 1

        # -------- cls_token --------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # -----------------------------------------------------------------
        # ViViT_MAE Encoder

        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dims))
        self.drop_after_encoder_pos = nn.Dropout(p=self.dropout_p)

        self.encoder_time_embed = nn.Parameter(
            torch.zeros(1, self.num_frames // self.tube_size + 1, self.embed_dims))
        self.drop_after_encoder_time = nn.Dropout(p=self.dropout_p)

        self.spatial_encoder = nn.ModuleList([
            BasicTransformerBlock(embed_dims=self.embed_dims,
                                  num_heads=self.num_heads,
                                  num_frames=self.num_frames,
                                  hidden_channels=self.embed_dims * 4,
                                  operator_order=['self_attn', 'ffn'],
                                  norm_layer=norm_layer,
                                  act_layer=act_layer,
                                  num_layers=self.num_layers,
                                  dpr=self.dropout_p)
            for _ in range(self.spatial_depth)])

        self.temporal_encoder = nn.ModuleList([
            BasicTransformerBlock(embed_dims=self.embed_dims,
                                  num_heads=self.num_heads,
                                  num_frames=self.num_frames,
                                  hidden_channels=self.embed_dims * 4,
                                  operator_order=['self_attn', 'ffn'],
                                  norm_layer=norm_layer,
                                  act_layer=act_layer,
                                  num_layers=self.num_layers,
                                  dpr=self.dropout_p)
            for _ in range(self.temporal_depth)])

        self.encoder_norm = norm_layer(self.embed_dims, eps=1e-6)

        # -----------------------------------------------------------------
        # ViViT_MAE Decoder

        self.decoder_embed = nn.Linear(self.embed_dims, self.embed_dims, bias=True)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dims))
        self.drop_after_decoder_pos = nn.Dropout(p=self.dropout_p)

        self.spatial_decoder = nn.ModuleList([
            BasicTransformerBlock(embed_dims=self.embed_dims,
                                  num_heads=self.num_heads,
                                  num_frames=self.num_frames,
                                  hidden_channels=self.embed_dims * 4,
                                  operator_order=['self_attn', 'ffn'],
                                  norm_layer=norm_layer,
                                  act_layer=act_layer,
                                  num_layers=self.num_layers,
                                  dpr=self.dropout_p)
            for _ in range(self.spatial_depth)])

        self.decoder_norm = norm_layer(self.embed_dims, eps=1e-6)
        self.decoder_pred = nn.Linear(
            self.embed_dims, self.patch_size ** 2 * in_channels, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self._init_weights_and_pretrain()

    def _load_pretrained(self, new_dict: dict, source: str):
        """Load pretrained weights with strict=False, logging missing/unexpected keys.

        Pretrained image models typically have fewer layers than the full
        spatial_depth + temporal_depth of this model, so layers beyond the
        pretrained split_point retain their ``_init_weights`` initialization.
        """
        result = self.load_state_dict(new_dict, strict=False)
        if result.missing_keys:
            print(f"[{source}] {len(result.missing_keys)} keys not in pretrained dict "
                  f"(keeping random init), e.g.: {result.missing_keys[:3]}")
        if result.unexpected_keys:
            print(f"[{source}] {len(result.unexpected_keys)} unexpected keys ignored, "
                  f"e.g.: {result.unexpected_keys[:3]}")

    @torch.no_grad()
    def _init_weights_and_pretrain(self):
        """Initialize all parameters, then optionally load pretrained weights."""
        self.apply(self._init_weights)

        trunc_normal_(self.patch_embed.proj.weight, std=.02)
        trunc_normal_(self.patch_embed.proj.bias, std=.02)
        trunc_normal_(self.encoder_pos_embed, std=.02)
        trunc_normal_(self.encoder_time_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        if self.mae_pretrain:
            model_name = "facebook/vit-mae-base" if self.arch == 'base' else "facebook/vit-mae-large"
            model = AutoModelForPreTraining.from_pretrained(model_name)
            old_state_dict = model.state_dict()
            new_dict = get_vit_dict_mae(
                old_state_dict=old_state_dict,
                tube_size=self.tube_size,
                embed_dims=self.embed_dims,
                num_frames=self.num_frames)
            self._load_pretrained(new_dict, source="MAE-ViT")

        elif self.vit_pretrain:
            model_name = "google/vit-base-patch16-224" if self.arch == 'base' else "google/vit-large-patch16-224"
            model = AutoModelForImageClassification.from_pretrained(model_name)
            old_state_dict = model.state_dict()
            new_dict = get_vit_dict_mae(
                old_state_dict=old_state_dict,
                tube_size=self.tube_size,
                embed_dims=self.embed_dims,
                num_frames=self.num_frames)
            self._load_pretrained(new_dict, source="ViT")

        elif self.clip_pretrain:
            model_name = "openai/clip-vit-base-patch32" if self.arch == 'base' else "openai/clip-vit-large-patch14"
            model = AutoModelForImageClassification.from_pretrained(model_name)
            old_state_dict = model.state_dict()
            new_dict = get_clip_dict_mae(
                old_state_dict=old_state_dict,
                tube_size=self.tube_size,
                embed_dims=self.embed_dims,
                num_frames=self.num_frames,
                num_patches=self.num_patches,
                in_channels=self.in_channels,
                patch_size=self.patch_size)
            self._load_pretrained(new_dict, source="CLIP")

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
        # Use expand instead of repeat for the index tensor (memory-efficient view)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patchify(self, imgs: Tensor) -> Tensor:
        """Convert video frames to patch sequences.

        Args:
            imgs: Input of shape (N, C, D, H, W).

        Returns:
            Patches of shape (N, D*H'*W', patch_size**2 * C).
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0

        # Use einops.rearrange for clarity instead of manual reshape + einsum
        x = rearrange(imgs, 'n c d (h p1) (w p2) -> n (h w d) (p1 p2 c)', p1=p, p2=p)
        return x

    def forward_encoder(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = x.float()
        x = x.permute(0, 2, 1, 3, 4)
        b = x.shape[0]
        t = self.num_frames // self.tube_size

        # -------- patch_embed --------
        x = self.patch_embed(x)

        # -------- pos_embed (skip cls position at index 0) --------
        x = x + self.encoder_pos_embed[:, 1:, :]
        x = self.drop_after_encoder_pos(x)

        # masking: length -> length * (1 - mask_ratio)
        x, mask, ids_restore = self.random_masking(x)
        # Tile mask across tube_size and rearrange to (batch, total_patches)
        mask = rearrange(
            mask.repeat(1, self.tube_size),
            '(b1 b2) p -> b1 (b2 p)', b1=b, b2=t)

        # -------- spatial cls_token (expand is a memory-efficient view) --------
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        spatial_cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((spatial_cls_tokens, x), dim=1)

        # -------- spatial transformer --------
        for block in self.spatial_encoder:
            x = block(x)
        x = self.encoder_norm(x)

        # -------- time_embed --------
        temporal_cls_tokens = x[:b, 0, :].unsqueeze(1)
        x = rearrange(x, '(b t) p d -> b t p d', b=b)
        x = reduce(x, 'b t p d -> b t d', 'mean')
        x = torch.cat((temporal_cls_tokens, x), dim=1)

        x = x + self.encoder_time_embed
        x = self.drop_after_encoder_time(x)

        # -------- temporal transformer --------
        for block in self.temporal_encoder:
            x = block(x)
        x = self.encoder_norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        b = x.shape[0]
        t = self.num_frames // self.tube_size

        # -------- embed tokens --------
        x = self.decoder_embed(x)

        # Replicate each time-step feature into spatial tokens
        # _num_visible = len_keep + 1 (cls-equivalent position)
        x = repeat(x, 'b t d -> b t p d', p=self._num_visible)
        x = rearrange(x, 'b t p d -> (b t) p d')
        # First b rows → temporal cls (repeated spatially)
        # Remaining b*t rows → one per (batch, time-step)

        # Append mask tokens to restore full spatial sequence
        num_mask_tokens = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.expand(x.shape[0] - b, num_mask_tokens, -1)
        x_ = torch.cat([x[b:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))

        # Re-attach spatial & temporal cls tokens
        spatial_cls_tokens = x[b:, :1, :]
        # expand is a memory-efficient view (no data copy) vs repeat
        temporal_cls_tokens = x[:b, :1, :].expand(-1, self._num_spatial_with_cls, -1)
        x = torch.cat((spatial_cls_tokens, x_), dim=1)
        x = torch.cat((temporal_cls_tokens, x), dim=0)

        # -------- add pos embed --------
        x = x + self.decoder_pos_embed
        x = self.drop_after_decoder_pos(x)

        # -------- spatial Transformer decoder --------
        for blk in self.spatial_decoder:
            x = blk(x)
        x = self.decoder_norm(x)

        # -------- predictor projection --------
        x = self.decoder_pred(x)

        # Remove temporal cls rows and spatial cls column, then tile for tube_size
        x = x[b:, 1:, :]
        x = x.repeat(1, self.tube_size, 1)
        x = rearrange(x, '(b1 b2) p f -> b1 (b2 p) f', b1=b, b2=t)

        return x

    def forward_loss(self, imgs: Tensor, pred: Tensor, mask: Tensor) -> Tensor:
        """Compute MSE reconstruction loss on masked patches.

        Args:
            imgs: Original video [N, C, D, H, W].
            pred: Predicted patches [N, L, p*p*C].
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
        latent, mask, ids_restore = self.forward_encoder(x)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask
