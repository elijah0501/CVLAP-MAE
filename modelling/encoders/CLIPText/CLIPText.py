import os
from collections import OrderedDict

import torch
from torch import nn

from data.kinetics.kinetics_dataloader import kinetics_dataloader
from utils.clip_utils import get_clip_text_dict
from utils.essential import get_cfg

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TextEncoder(nn.Module):

    ARCH_CONFIGS = {
        'base': dict(transformer_width=512, transformer_heads=8, transformer_layers=12),
        'large': dict(transformer_width=768, transformer_heads=12, transformer_layers=12),
    }

    def __init__(self,
                 embed_dim,
                 # text
                 context_length,
                 vocab_size,
                 transformer_width,
                 transformer_heads,
                 transformer_layers,
                 pretrained,
                 arch='base'
                 ):
        super(TextEncoder, self).__init__()

        # -------- architecture override --------
        assert arch in self.ARCH_CONFIGS, f"Unsupported arch '{arch}'. Choose from {list(self.ARCH_CONFIGS.keys())}"
        arch_cfg = self.ARCH_CONFIGS[arch]
        transformer_width = arch_cfg['transformer_width']
        transformer_heads = arch_cfg['transformer_heads']
        transformer_layers = arch_cfg['transformer_layers']

        self.arch = arch
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers

        self.pretrained = pretrained

        self.transformer = Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, self.embed_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        if self.pretrained:
            # Select arch-appropriate checkpoint
            pretrained_path = self.pretrained
            if self.arch == 'large':
                pretrained_path = pretrained_path.replace('ViT-B-16.pt', 'ViT-L-14.pt').replace('ViT-B-32.pt', 'ViT-L-14.pt')
            old_state_dict = torch.jit.load(pretrained_path).state_dict()
            new_state_dict = get_clip_text_dict(old_state_dict,
                                                transformer_width=self.transformer_width,
                                                embed_dim=self.embed_dim)

            # Handle text_projection shape mismatch when embed_dim differs
            # from pretrained (e.g. base checkpoint 512→512, but we want 512→768)
            pretrained_proj = new_state_dict.get('text_projection')
            if pretrained_proj is not None and pretrained_proj.shape != self.text_projection.shape:
                print(f'[TextEncoder] text_projection shape mismatch: '
                      f'pretrained {pretrained_proj.shape} vs model {self.text_projection.shape}. '
                      f'Loading other weights only; text_projection will use random init.')
                del new_state_dict['text_projection']

            self.load_state_dict(new_state_dict, strict=False)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        text = text.long()
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x.add_(self.positional_embedding)
        x = self.transformer(x)
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, text):
        text_features = self.encode_text(text)
        return text_features
