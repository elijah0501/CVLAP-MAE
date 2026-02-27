import torch.nn as nn
import torch.nn.functional as F

from modelling.encoders.ViViT.ViViT import ViViT


class CLS_ViViT(nn.Module):
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
                 head_features,
                 num_classes,
                 arch='base',
                 **kwargs):
        super(CLS_ViViT, self).__init__()

        self.encoder = ViViT(input_size=input_size,
                  patch_size=patch_size,
                  tube_size=tube_size,
                  in_channels=in_channels,
                  embed_dims=embed_dims,
                  spatial_depth=spatial_depth,
                  temporal_depth=temporal_depth,
                  num_heads=num_heads,
                  num_frames=num_frames,
                  dropout_p=dropout_p,
                  return_cls_token=return_cls_token,
                  pretrained=pretrained,
                  num_layers=num_layers,
                  mae_pretrain=mae_pretrain,
                  vit_pretrain=vit_pretrain,
                  clip_pretrain=clip_pretrain,
                  arch=arch)

        # Classification head
        actual_embed_dims = self.encoder.embed_dims
        self.head_fc1 = nn.Linear(actual_embed_dims, head_features)
        self.head_fc2 = nn.Linear(head_features, num_classes)

    def forward(self, x):
        x = self.encoder(x)

        # Classification
        x = F.relu(self.head_fc1(x))
        x = self.head_fc2(x)
        return x
