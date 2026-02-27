import numpy as np
import torch
from torch import nn


# Custom PatchEmbed to match the input shape
class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = None
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w, cls_token=False, dist_token=False):
    """
    grid_size_h: int, grid height (number of patches along height)
    grid_size_w: int, grid width (number of patches along width)
    return:
    pos_embed: [grid_size_h*grid_size_w, embed_dim] or [1+grid_size_h*grid_size_w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and dist_token:
        pos_embed = np.concatenate([np.zeros([2, embed_dim]), pos_embed], axis=0)
    elif cls_token or dist_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed



def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    grid: [2, 1, grid_size_h, grid_size_w]
    return:
    pos_embed: [grid_size_h*grid_size_w, embed_dim]
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return pos_embed



def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.flatten()  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



def get_vit_dict(old_state_dict, embed_dims):

    new_dict = {}
    for old_key, value in old_state_dict.items():
        if 'module.v.cls_token' in old_key:
            new_dict['cls_token'] = value
        elif 'module.v.pos_embed' in old_key:
            new_dict['pos_embed'] = nn.Parameter(torch.zeros(1, 512, embed_dims))
        elif 'module.v.dist_token' in old_key:
            new_dict['dist_token'] = value
        elif 'module.v.patch_embed.proj' in old_key:
            k = old_key.replace('module.v.patch_embed', 'patch_embed')
            new_dict[k] = value
        elif 'module.v.norm' in old_key:
            k = old_key.replace('module.v.norm', 'norm')
            new_dict[k] = value
        elif 'module.v.blocks' in old_key:
            k = old_key.replace('module.v.blocks', 'blocks')
            new_dict[k] = value

    return new_dict


def get_vit_dict_mae(old_state_dict, embed_dims, num_patches, decoder_embed_dim, pred_dims):

    new_dict = {}
    for old_key, value in old_state_dict.items():
        if 'module.v.cls_token' in old_key:
            new_dict['cls_token'] = value
        elif 'module.v.pos_embed' in old_key:
            new_dict['pos_embed'] = nn.Parameter(torch.zeros(1, 512, embed_dims))
        elif 'module.v.dist_token' in old_key:
            new_dict['dist_token'] = value
        elif 'module.v.patch_embed.proj' in old_key:
            k = old_key.replace('module.v.patch_embed', 'patch_embed')
            new_dict[k] = value
        elif 'module.v.norm' in old_key:
            encoder_old_key = old_key.replace('module.v.norm', 'encoder_norm')
            decoder_old_key = old_key.replace('module.v.norm', 'decoder_norm')
            new_dict[encoder_old_key] = value
            new_dict[decoder_old_key] = value
        elif 'module.v.blocks' in old_key:
            encoder_old_key = old_key.replace('module.v.blocks', 'encoder_blocks')
            decoder_old_key = old_key.replace('module.v.blocks', 'decoder_blocks')
            new_dict[encoder_old_key] = value
            new_dict[decoder_old_key] = value

    new_dict["mask_token"] = torch.zeros(1, 1, embed_dims)
    new_dict["decoder_embed.weight"] = torch.zeros(embed_dims, embed_dims)
    new_dict["decoder_embed.bias"] = torch.zeros(embed_dims)
    new_dict["decoder_pred.weight"] = torch.zeros(pred_dims, decoder_embed_dim)
    new_dict["decoder_pred.bias"] = torch.zeros(pred_dims)
    new_dict["decoder_pos_embed"] = torch.zeros(1, num_patches + 1, decoder_embed_dim)

    return new_dict
