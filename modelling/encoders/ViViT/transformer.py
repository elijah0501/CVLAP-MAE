import numpy as np
import torch
from torch import nn


class BasicTransformerBlock(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 hidden_channels,
                 operator_order,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 num_layers=2,
                 dpr=0,
                 ):

        super().__init__()
        self.attentions = nn.ModuleList([])
        self.ffns = nn.ModuleList([])

        for i, operator in enumerate(operator_order):
            if operator == 'self_attn':
                self.attentions.append(
                    MultiheadAttentionWithPreNorm(
                        embed_dims=embed_dims,
                        num_heads=num_heads,
                        batch_first=True,
                        norm_layer=nn.LayerNorm,
                        layer_drop=dict(type=DropPath, dropout_p=dpr)))
            elif operator == 'ffn':
                self.ffns.append(
                    FFNWithPreNorm(
                        embed_dims=embed_dims,
                        hidden_channels=hidden_channels,
                        num_layers=num_layers,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        layer_drop=dict(type=DropPath, dropout_p=dpr)))
            else:
                raise TypeError(f'Unsupported operator type {operator}')

    def forward(self, x):
        for layer in self.attentions:
            x = layer(x)
        for layer in self.ffns:
            x = layer(x)
        return x


class DropPath(nn.Module):

    def __init__(self, dropout_p=None):
        super(DropPath, self).__init__()
        self.dropout_p = dropout_p

    def forward(self, x):
        return self.drop_path(x, self.dropout_p, self.training)

    def drop_path(self, x, dropout_p=0., training=False):
        if dropout_p == 0. or not training:
            return x
        keep_prob = 1 - dropout_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape).type_as(x)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class MultiheadAttentionWithPreNorm(nn.Module):
    """Implements MultiheadAttention with residual connection.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        norm_layer (class): Class name for normalization layer. Defaults to
            nn.LayerNorm.
        layer_drop (obj:`ConfigDict`): The layer_drop used
            when adding the shortcut.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 norm_layer=nn.LayerNorm,
                 layer_drop=dict(type=DropPath, dropout_p=0.),
                 batch_first=False,
                 **kwargs):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.norm = norm_layer(embed_dims)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          batch_first=batch_first, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        dropout_p = layer_drop.pop('dropout_p')
        layer_drop = layer_drop.pop('type')
        self.layer_drop = layer_drop(dropout_p) if layer_drop else nn.Identity()

    def forward(self,
                query,
                key=None,
                value=None,
                residual=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        residual = query

        query = self.norm(query)
        attn_out = self.attn(
            query=query,
            key=query,
            value=query,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        new_query = residual + self.layer_drop(self.proj_drop(attn_out))
        return new_query


class FFNWithPreNorm(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        hidden_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_layers (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_layer (dict, optional): The activation layer for FFNs.
            Default: nn.GELU
        norm_layer (class): Class name for normalization layer. Defaults to
            nn.LayerNorm.
        dropout_p (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        layer_drop (obj:`ConfigDict`): The layer_drop used
            when adding the shortcut.
    """

    def __init__(self,
                 embed_dims=256,
                 hidden_channels=1024,
                 num_layers=2,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 dropout_p=0.,
                 layer_drop=None,
                 **kwargs):
        super().__init__()
        assert num_layers >= 2, 'num_layers should be no less ' \
                                f'than 2. got {num_layers}.'
        self.embed_dims = embed_dims
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.norm = norm_layer(embed_dims)
        layers = []
        in_channels = embed_dims
        for _ in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_channels),
                    act_layer(),
                    nn.Dropout(dropout_p)))
            in_channels = hidden_channels
        layers.append(nn.Linear(hidden_channels, embed_dims))
        layers.append(nn.Dropout(dropout_p))
        self.layers = nn.ModuleList(layers)

        if layer_drop:
            dropout_p = layer_drop.pop('dropout_p')
            layer_drop = layer_drop.pop('type')
            self.layer_drop = layer_drop(dropout_p)
        else:
            self.layer_drop = nn.Identity()

    def forward(self, x):
        residual = x

        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)

        return residual + self.layer_drop(x)
