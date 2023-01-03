import timm
import torch

from models.backbone.base import BaseNet


class SwinViTBase(BaseNet):
    name = 'swin_base_patch4_window7_224'

    def __init__(self, pretrained=True, **kwargs):
        super(SwinViTBase, self).__init__()

        model = timm.create_model(self.name, pretrained=pretrained)

        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.layers = model.layers
        self.norm = model.norm
        self.avgpool = model.avgpool
        self.head = model.head

        self.features_size = model.head.in_features

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.layers(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def classify(self, x):
        x = self.forward(x)
        x = self.head(x)
        return x


class SwinViTTiny(SwinViTBase):
    name = 'swin_tiny_patch4_window7_224'


class SwinViTSmall(SwinViTBase):
    name = 'swin_small_patch4_window7_224'
