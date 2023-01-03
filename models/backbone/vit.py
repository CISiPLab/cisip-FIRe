import collections.abc
import math
import os
from itertools import repeat

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.vit import ViTModel

from models.backbone.base import BaseNet


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
ROOTDIR = os.environ.get('ROOTDIR', '.')


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class ViTBase(BaseNet):
    name = 'vit_base_patch16_224'

    def __init__(self, pretrained=True, **kwargs):
        super(ViTBase, self).__init__()

        model = timm.create_model(self.name, pretrained=pretrained,
                                  drop_rate=kwargs.get('drop_rate', 0.),
                                  attn_drop_rate=kwargs.get('attn_drop_rate', 0.),
                                  drop_path_rate=kwargs.get('drop_path_rate', 0.))

        self.embed_dim = model.embed_dim
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.blocks = model.blocks
        self.norm = model.norm
        self.pre_logits = nn.Identity()
        self.head = model.head

        self.features_size = model.head.in_features
        self.pool_method = kwargs.get('pool_method', 'cls_token')
        print('pool_method:', self.pool_method)

        self.replace_patch_embed()

    def replace_patch_embed(self):
        # original vit cannot support other input sizes
        patch_embed = PatchEmbed(self.patch_embed.img_size,
                                 self.patch_embed.patch_size,
                                 3,
                                 self.embed_dim)
        patch_embed.load_state_dict(self.patch_embed.state_dict())
        self.patch_embed = patch_embed

    def interpolate_pos_embedding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[1]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x, with_feat_map=False):
        b, _, h, w = x.size()
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.interpolate_pos_embedding(x, w, h))
        x = self.blocks(x)
        x = self.norm(x)

        if with_feat_map:
            nh = h // self.patch_embed.patch_size[0]
            nw = w // self.patch_embed.patch_size[1]

            return self.pre_logits(x[:, 0]), x[:, 1:].reshape(b, nh, nw, -1).permute(0, 3, 1, 2)
        else:
            if self.pool_method == 'cls_token':
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 1:, :].mean(dim=1)

    def classify(self, x):
        x = self.forward(x)
        x = self.head(x)
        return x


class ViTTiny(ViTBase):
    name = 'vit_tiny_patch16_224'


class ViTSmall(ViTBase):
    name = 'vit_small_patch16_224'


class HuggingFaceViT(nn.Module):
    NAMES = {
        'base_patch32': 'google/vit-base-patch32-224-in21k',
        'base_patch16': 'google/vit-base-patch16-224-in21k'
    }

    def __init__(self, name='google/vit-base-patch32-224-in21k', **kwargs):
        super().__init__()

        self.model = ViTModel.from_pretrained(name)  # type: ViTModel

    def forward(self, image):
        return self.model(image)[1]


def _attn_forward(attn_module, x):
    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
    attn = attn.softmax(dim=-1)
    attn = attn_module.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x, attn


@torch.no_grad()
def get_attention_and_outputs(vit_model: ViTBase, x: torch.Tensor):
    outputs = {}

    b, _, h, w = x.size()
    x = vit_model.patch_embed(x)
    outputs['patch_embed'] = x

    cls_token = vit_model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)

    x = vit_model.pos_drop(x + vit_model.interpolate_pos_embedding(x, w, h))
    outputs['pos_embed'] = x

    for bidx, block in enumerate(vit_model.blocks):
        z, attn = _attn_forward(block.attn, block.norm1(x))
        x = x + block.drop_path(z)
        x = x + block.drop_path(block.mlp(block.norm2(x)))
        outputs[f'block_{bidx}_attn'] = attn
        outputs[f'block_{bidx}_x'] = x

    x = vit_model.norm(x)
    outputs['output'] = x

    return outputs
