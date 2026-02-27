import torch
from einops import rearrange, repeat
from timm.models.layers import to_2tuple
from torch import nn


class PatchEmbed(nn.Module):
    """video tube Patch Embedding.

    Args:
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        tube_size (int): Size of temporal field of one 3D patch.
        in_channels (int): Channel num of input features. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        conv_type (str): Type for convolution layer. Defaults to 'Conv2d'.
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 tube_size,
                 in_channels,
                 embed_dims):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)

        num_patches = \
            (self.img_size[1] // self.patch_size[1]) * \
            (self.img_size[0] // self.patch_size[0])

        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            in_channels,
            embed_dims,
            kernel_size=(tube_size, patch_size, patch_size),
            stride=(tube_size, patch_size, patch_size))

        self.init_weights(self.proj)

    @staticmethod
    def init_weights(module):
        if hasattr(module, 'weight') and module.weight is not None:
            kaiming_init_(module.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(module, 'bias') and module.bias is not None:
            constant_init_(module.bias, constant_value=0)

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> b c t h w')
        x = self.proj(x)
        x = rearrange(x, 'b c t h w -> (b t) (h w) c')
        return x


@torch.no_grad()
def constant_init_(tensor, constant_value=0):
    nn.init.constant_(tensor, constant_value)


@torch.no_grad()
def kaiming_init_(tensor,
                  a=0,
                  mode='fan_out',
                  nonlinearity='relu',
                  distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            tensor, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            tensor, a=a, mode=mode, nonlinearity=nonlinearity)


def get_vit_dict(old_state_dict, tube_size, embed_dims, num_frames, temporal_depth=4):
    new_dict = {}
    qkv_dim = 3 * embed_dims
    # Compute split point dynamically from pretrained model depth
    pretrained_layers = max(
        (int(k.split('.')[3]) for k in old_state_dict if 'vit.encoder.layer' in k), default=-1) + 1
    split_point = pretrained_layers - temporal_depth

    # Iterate through the parameter dict
    for old_key, value in old_state_dict.items():
        if 'vit.embeddings.patch_embeddings.projection.weight' in old_key:
            value = repeat(value, 'd c h w -> d c t h w', t=tube_size) / tube_size
            new_dict["patch_embed.proj.weight"] = value

        elif 'vit.embeddings.patch_embeddings.projection.bias' in old_key:
            new_dict["patch_embed.proj.bias"] = value

        elif 'cls_token' in old_key:
            new_dict["cls_token"] = value

        elif 'position_embeddings' in old_key:
            new_dict["pos_embed"] = value

        elif 'vit.layernorm.weight' in old_key:
            new_dict["norm.weight"] = value

        elif 'vit.layernorm.bias' in old_key:
            new_dict["norm.bias"] = value

        elif 'vit.encoder.layer' in old_key:
            layer_num = int(old_key.split('.')[3])
            if layer_num < split_point:
                encoder_old_key = old_key.replace(f'vit.encoder.layer.{layer_num}', f'spatial_encoder.{layer_num}')
                # in_proj_weight & in_proj_bias
                in_proj_weight_key = f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_weight"
                if in_proj_weight_key not in new_dict:
                    new_dict[in_proj_weight_key] = torch.zeros(qkv_dim, embed_dims)
                in_proj_bias_key = f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_bias"
                if in_proj_bias_key not in new_dict:
                    new_dict[in_proj_bias_key] = torch.zeros(qkv_dim)
            else:
                encoder_old_key = old_key.replace(f'vit.encoder.layer.{layer_num}',
                                                  f'temporal_encoder.{layer_num - split_point}')
                # in_proj_weight & in_proj_bias
                in_proj_weight_key = f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_weight"
                if in_proj_weight_key not in new_dict:
                    new_dict[in_proj_weight_key] = torch.zeros(qkv_dim, embed_dims)
                in_proj_bias_key = f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_bias"
                if in_proj_bias_key not in new_dict:
                    new_dict[in_proj_bias_key] = torch.zeros(qkv_dim)

            if 'layernorm_before' in encoder_old_key:
                k = encoder_old_key.replace(f'layernorm_before', f'attentions.0.norm')
                new_dict[k] = value

            elif 'layernorm_after' in encoder_old_key:
                k = encoder_old_key.replace(f'layernorm_after', f'ffns.0.norm')
                new_dict[k] = value

            elif 'attention.output.dense' in encoder_old_key:
                k = encoder_old_key.replace(f'attention.output.dense', f'attentions.0.attn.out_proj')
                new_dict[k] = value

            elif 'intermediate.dense' in encoder_old_key:
                k = encoder_old_key.replace(f'intermediate.dense', f'ffns.0.layers.0.0')
                new_dict[k] = value

            elif 'output.dense' in encoder_old_key:
                k = encoder_old_key.replace(f'output.dense', f'ffns.0.layers.1')
                new_dict[k] = value

    # Add a zero-initialized time_embed
    new_dict["time_embed"] = torch.zeros(1, num_frames // tube_size + 1, embed_dims)

    return new_dict


def get_clip_dict(old_state_dict, tube_size, embed_dims, num_frames, num_patches, in_channels, patch_size, temporal_depth=4):
    qkv_dim = 3 * embed_dims
    # Compute split point dynamically from pretrained model depth
    pretrained_layers = max(
        (int(k.split('.')[3]) for k in old_state_dict if 'vision_model.encoder.layers' in k), default=-1) + 1
    split_point = pretrained_layers - temporal_depth

    new_dict = {"pos_embed": torch.zeros(1, num_patches + 1, embed_dims),
                "time_embed": torch.zeros(1, num_frames // tube_size + 1, embed_dims),
                "cls_token": torch.zeros(1, 1, embed_dims),
                "patch_embed.proj.weight": torch.zeros(embed_dims, in_channels, tube_size, patch_size, patch_size),
                "patch_embed.proj.bias": torch.zeros(embed_dims)}

    for old_key, value in old_state_dict.items():
        if 'vision_model.encoder.layers' in old_key:
            layer_num = int(old_key.split('.')[3])
            if layer_num < split_point:
                encoder_old_key = old_key.replace(f'vision_model.encoder.layers.{layer_num}',
                                                  f'spatial_encoder.{layer_num}')
                # in_proj_weight & in_proj_bias
                in_proj_weight_key = f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_weight"
                if in_proj_weight_key not in new_dict:
                    new_dict[in_proj_weight_key] = torch.zeros(qkv_dim, embed_dims)
                in_proj_bias_key = f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_bias"
                if in_proj_bias_key not in new_dict:
                    new_dict[in_proj_bias_key] = torch.zeros(qkv_dim)
            else:
                encoder_old_key = old_key.replace(f'vision_model.encoder.layers.{layer_num}',
                                                  f'temporal_encoder.{layer_num - split_point}')
                # in_proj_weight & in_proj_bias
                in_proj_weight_key = f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_weight"
                if in_proj_weight_key not in new_dict:
                    new_dict[in_proj_weight_key] = torch.zeros(qkv_dim, embed_dims)
                in_proj_bias_key = f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_bias"
                if in_proj_bias_key not in new_dict:
                    new_dict[in_proj_bias_key] = torch.zeros(qkv_dim)

            if 'layer_norm1' in encoder_old_key:
                k = encoder_old_key.replace(f'layer_norm1', f'attentions.0.norm')
                new_dict[k] = value

            elif 'layer_norm2' in encoder_old_key:
                k = encoder_old_key.replace(f'layer_norm2', f'ffns.0.norm')
                new_dict[k] = value

            elif 'out_proj' in encoder_old_key:
                k = encoder_old_key.replace(f'self_attn.out_proj', f'attentions.0.attn.out_proj')
                new_dict[k] = value

            elif 'mlp.fc1' in encoder_old_key:
                k = encoder_old_key.replace(f'mlp.fc1', f'ffns.0.layers.0.0')
                new_dict[k] = value

            elif 'mlp.fc2' in encoder_old_key:
                k = encoder_old_key.replace(f'mlp.fc2', f'ffns.0.layers.1')
                new_dict[k] = value

        elif 'vision_model.post_layernorm.weight' in old_key:
            new_dict["norm.weight"] = value

        elif 'vision_model.post_layernorm.bias' in old_key:
            new_dict["norm.bias"] = value

    return new_dict


def get_vit_dict_mae(old_state_dict, tube_size, embed_dims, num_frames, temporal_depth=4):
    new_dict = {}
    qkv_dim = 3 * embed_dims
    # Compute split point dynamically from pretrained model depth
    pretrained_layers = max(
        (int(k.split('.')[3]) for k in old_state_dict if 'vit.encoder.layer' in k), default=-1) + 1
    split_point = pretrained_layers - temporal_depth

    for old_key, value in old_state_dict.items():
        if 'vit.embeddings.patch_embeddings.projection.weight' in old_key:
            value = repeat(value, 'd c h w -> d c t h w', t=tube_size) / tube_size
            new_dict["patch_embed.proj.weight"] = value

        elif 'vit.embeddings.patch_embeddings.projection.bias' in old_key:
            new_dict["patch_embed.proj.bias"] = value

        elif 'cls_token' in old_key:
            new_dict["cls_token"] = value

        elif 'position_embeddings' in old_key:
            new_dict["encoder_pos_embed"] = value
            new_dict["decoder_pos_embed"] = value

        elif 'vit.layernorm.weight' in old_key:
            new_dict["encoder_norm.weight"] = value
            new_dict["decoder_norm.weight"] = value

        elif 'vit.layernorm.bias' in old_key:
            new_dict["encoder_norm.bias"] = value
            new_dict["decoder_norm.bias"] = value

        elif 'vit.encoder.layer' in old_key:
            layer_num = int(old_key.split('.')[3])
            decoder_old_key = None
            if layer_num < split_point:
                encoder_old_key = old_key.replace(f'vit.encoder.layer.{layer_num}', f'spatial_encoder.{layer_num}')
                decoder_old_key = old_key.replace(f'vit.encoder.layer.{layer_num}', f'spatial_decoder.{layer_num}')
                # in_proj_weight & in_proj_bias
                # encoder
                in_proj_weight_key = f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_weight"
                if in_proj_weight_key not in new_dict:
                    new_dict[in_proj_weight_key] = torch.zeros(qkv_dim, embed_dims)
                in_proj_bias_key = f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_bias"
                if in_proj_bias_key not in new_dict:
                    new_dict[in_proj_bias_key] = torch.zeros(qkv_dim)
                # decoder
                decoder_in_proj_weight_key = f"spatial_decoder.{layer_num}.attentions.0.attn.in_proj_weight"
                if decoder_in_proj_weight_key not in new_dict:
                    new_dict[decoder_in_proj_weight_key] = torch.zeros(qkv_dim, embed_dims)
                decoder_in_proj_bias_key = f"spatial_decoder.{layer_num}.attentions.0.attn.in_proj_bias"
                if decoder_in_proj_bias_key not in new_dict:
                    new_dict[decoder_in_proj_bias_key] = torch.zeros(qkv_dim)
            else:
                encoder_old_key = old_key.replace(f'vit.encoder.layer.{layer_num}',
                                                  f'temporal_encoder.{layer_num - split_point}')
                # in_proj_weight & in_proj_bias
                # encoder
                in_proj_weight_key = f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_weight"
                if in_proj_weight_key not in new_dict:
                    new_dict[in_proj_weight_key] = torch.zeros(qkv_dim, embed_dims)
                in_proj_bias_key = f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_bias"
                if in_proj_bias_key not in new_dict:
                    new_dict[in_proj_bias_key] = torch.zeros(qkv_dim)

            if 'layernorm_before' in encoder_old_key:
                k = encoder_old_key.replace(f'layernorm_before', f'attentions.0.norm')
                new_dict[k] = value

            elif 'layernorm_after' in encoder_old_key:
                k = encoder_old_key.replace(f'layernorm_after', f'ffns.0.norm')
                new_dict[k] = value

            elif 'attention.output.dense' in encoder_old_key:
                k = encoder_old_key.replace(f'attention.output.dense', f'attentions.0.attn.out_proj')
                new_dict[k] = value

            elif 'intermediate.dense' in encoder_old_key:
                k = encoder_old_key.replace(f'intermediate.dense', f'ffns.0.layers.0.0')
                new_dict[k] = value

            elif 'output.dense' in encoder_old_key:
                k = encoder_old_key.replace(f'output.dense', f'ffns.0.layers.1')
                new_dict[k] = value

            if decoder_old_key is not None:
                if 'layernorm_before' in decoder_old_key:
                    k = decoder_old_key.replace(f'layernorm_before', f'attentions.0.norm')
                    new_dict[k] = value

                elif 'layernorm_after' in decoder_old_key:
                    k = decoder_old_key.replace(f'layernorm_after', f'ffns.0.norm')
                    new_dict[k] = value

                elif 'attention.output.dense' in decoder_old_key:
                    k = decoder_old_key.replace(f'attention.output.dense', f'attentions.0.attn.out_proj')
                    new_dict[k] = value

                elif 'intermediate.dense' in decoder_old_key:
                    k = decoder_old_key.replace(f'intermediate.dense', f'ffns.0.layers.0.0')
                    new_dict[k] = value

                elif 'output.dense' in decoder_old_key:
                    k = decoder_old_key.replace(f'output.dense', f'ffns.0.layers.1')
                    new_dict[k] = value

    new_dict["mask_token"] = torch.zeros(1, 1, embed_dims)
    new_dict["encoder_time_embed"] = torch.zeros(1, num_frames // tube_size + 1, embed_dims)
    new_dict["decoder_embed.weight"] = torch.zeros(embed_dims, embed_dims)
    new_dict["decoder_embed.bias"] = torch.zeros(embed_dims)
    new_dict["decoder_pred.weight"] = torch.zeros(embed_dims, embed_dims)
    new_dict["decoder_pred.bias"] = torch.zeros(embed_dims)

    return new_dict


def get_clip_dict_mae(old_state_dict, tube_size, embed_dims, num_frames, num_patches, in_channels, patch_size, temporal_depth=4):
    new_dict = {}
    qkv_dim = 3 * embed_dims
    # Compute split point dynamically from pretrained model depth
    pretrained_layers = max(
        (int(k.split('.')[3]) for k in old_state_dict if 'vision_model.encoder.layers' in k), default=-1) + 1
    split_point = pretrained_layers - temporal_depth

    for old_key, value in old_state_dict.items():
        if 'vision_model.encoder.layers' in old_key:
            layer_num = int(old_key.split('.')[3])
            decoder_old_key = None
            if layer_num < split_point:
                encoder_old_key = old_key.replace(f'vision_model.encoder.layers.{layer_num}',
                                                  f'spatial_encoder.{layer_num}')
                decoder_old_key = old_key.replace(f'vision_model.encoder.layers.{layer_num}',
                                                  f'spatial_decoder.{layer_num}')
                # in_proj_weight & in_proj_bias
                # encoder
                new_dict[f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_weight"] = torch.zeros(
                    qkv_dim, embed_dims)
                new_dict[f"spatial_encoder.{layer_num}.attentions.0.attn.in_proj_bias"] = torch.zeros(qkv_dim)
                # decoder
                new_dict[f"spatial_decoder.{layer_num}.attentions.0.attn.in_proj_weight"] = torch.zeros(
                    qkv_dim, embed_dims)
                new_dict[f"spatial_decoder.{layer_num}.attentions.0.attn.in_proj_bias"] = torch.zeros(qkv_dim)
            else:
                encoder_old_key = old_key.replace(f'vision_model.encoder.layers.{layer_num}',
                                                  f'temporal_encoder.{layer_num - split_point}')
                # in_proj_weight & in_proj_bias
                # only temporal_encoder
                new_dict[f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_weight"] = torch.zeros(
                    qkv_dim, embed_dims)
                new_dict[f"temporal_encoder.{layer_num - split_point}.attentions.0.attn.in_proj_bias"] = torch.zeros(qkv_dim)

            if 'layer_norm1' in encoder_old_key:
                k = encoder_old_key.replace(f'layer_norm1', f'attentions.0.norm')
                new_dict[k] = value

            elif 'layer_norm2' in encoder_old_key:
                k = encoder_old_key.replace(f'layer_norm2', f'ffns.0.norm')
                new_dict[k] = value

            elif 'out_proj' in encoder_old_key:
                k = encoder_old_key.replace(f'self_attn.out_proj', f'attentions.0.attn.out_proj')
                new_dict[k] = value

            elif 'mlp.fc1' in encoder_old_key:
                k = encoder_old_key.replace(f'mlp.fc1', f'ffns.0.layers.0.0')
                new_dict[k] = value

            elif 'mlp.fc2' in encoder_old_key:
                k = encoder_old_key.replace(f'mlp.fc2', f'ffns.0.layers.1')
                new_dict[k] = value

            if decoder_old_key is not None:
                if 'layer_norm1' in decoder_old_key:
                    k = decoder_old_key.replace(f'layer_norm1', f'attentions.0.norm')
                    new_dict[k] = value

                elif 'layer_norm2' in decoder_old_key:
                    k = decoder_old_key.replace(f'layer_norm2', f'ffns.0.norm')
                    new_dict[k] = value

                elif 'out_proj' in decoder_old_key:
                    k = decoder_old_key.replace(f'self_attn.out_proj', f'attentions.0.attn.out_proj')
                    new_dict[k] = value

                elif 'mlp.fc1' in decoder_old_key:
                    k = decoder_old_key.replace(f'mlp.fc1', f'ffns.0.layers.0.0')
                    new_dict[k] = value

                elif 'mlp.fc2' in decoder_old_key:
                    k = decoder_old_key.replace(f'mlp.fc2', f'ffns.0.layers.1')
                    new_dict[k] = value

        elif 'vision_model.post_layernorm.weight' in old_key:
            new_dict["decoder_norm.weight"] = value

        elif 'vision_model.post_layernorm.bias' in old_key:
            new_dict["decoder_norm.bias"] = value

    new_dict["encoder_pos_embed"] = torch.zeros(1, num_patches + 1, embed_dims)
    new_dict["encoder_time_embed"] = torch.zeros(1, num_frames // tube_size + 1, embed_dims)
    new_dict["decoder_pos_embed"] = torch.zeros(1, num_patches + 1, embed_dims)
    new_dict["mask_token"] = torch.zeros(1, 1, embed_dims)
    new_dict["cls_token"] = torch.zeros(1, 1, embed_dims)
    new_dict["patch_embed.proj.weight"] = torch.zeros(embed_dims, in_channels, tube_size, patch_size, patch_size)
    new_dict["patch_embed.proj.bias"] = torch.zeros(embed_dims)

    new_dict["encoder_norm.weight"] = torch.zeros(embed_dims)
    new_dict["encoder_norm.bias"] = torch.zeros(embed_dims)
    new_dict["decoder_embed.weight"] = torch.zeros(embed_dims, embed_dims)
    new_dict["decoder_embed.bias"] = torch.zeros(embed_dims)
    new_dict["decoder_pred.weight"] = torch.zeros(patch_size ** 2 * in_channels, embed_dims)
    new_dict["decoder_pred.bias"] = torch.zeros(patch_size ** 2 * in_channels)

    return new_dict
