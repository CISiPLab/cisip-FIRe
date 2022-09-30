import torch.nn as nn

from models.backbone.alexnet import AlexNetBackbone
from models.backbone.convnext import ConvNextBackbone
from models.backbone.linear import LinearBackbone
from models.backbone.mlp import MLPBackbone
from models.backbone.resnet import ResNetBackbone
from models.backbone.swinvit import SwinTransformerBackbone
from models.backbone.vgg import VGGBackbone
from models.backbone.vit import ViTBackbone
from models.layers.activation import SignHashLayer, StochasticBinaryLayer
from models.layers.bihalf import BiHalfLayer
from models.layers.zm import MeanOnlyBatchNorm


def get_backbone(backbone, nbit, nclass, pretrained, freeze_weight, **kwargs):
    if backbone == 'alexnet':
        return AlexNetBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                               freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'resnet18':
        return ResNetBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                              resnet_size='18', freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'resnet34':
        return ResNetBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                              resnet_size='34', freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'resnet50':
        return ResNetBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                              resnet_size='50', freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'resnet101':
        return ResNetBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                              resnet_size='101', freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'resnet152':
        return ResNetBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                              resnet_size='152', freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'vgg16':
        return VGGBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                           vgg_size='vgg16', freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'vgg16bn':
        return VGGBackbone(nbit=nbit, nclass=nclass, pretrained=pretrained,
                           vgg_size='vgg16bn', freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'linear':
        return LinearBackbone(nclass=nclass, nbit=nbit, **kwargs)
    elif backbone == 'mlp':
        return MLPBackbone(nclass=nclass, nbit=nbit, **kwargs)
    elif backbone == 'vit':
        return ViTBackbone(nbit=nbit, nclass=nclass, vit_name='vit_base_patch16_224',
                           pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'vittiny':
        return ViTBackbone(nbit=nbit, nclass=nclass, vit_name='vit_tiny_patch16_224',
                           pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'vitsmall':
        return ViTBackbone(nbit=nbit, nclass=nclass, vit_name='vit_small_patch16_224',
                           pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'swin':
        return SwinTransformerBackbone(nbit=nbit, nclass=nclass, vit_name='swin_base_patch4_window7_224',
                                       pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'swintiny':
        return SwinTransformerBackbone(nbit=nbit, nclass=nclass, vit_name='swin_tiny_patch4_window7_224',
                                       pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'swinsmall':
        return SwinTransformerBackbone(nbit=nbit, nclass=nclass, vit_name='swin_small_patch4_window7_224',
                                       pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'convnextbase':
        return ConvNextBackbone(nbit=nbit, nclass=nclass, vit_name='convnext_base_in22k',
                                pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    elif backbone == 'convnexttiny':
        return ConvNextBackbone(nbit=nbit, nclass=nclass, vit_name='convnext_base_in22k',
                                pretrained=pretrained, freeze_weight=freeze_weight, **kwargs)
    else:
        raise NotImplementedError('The backbone not implemented.')


def get_hash_fc_with_normalizations(in_features, nbit, bias, kwargs):
    output_choice = kwargs.get('hash_fc_output', 'identity')
    if output_choice == 'bn':  # kwargs.get('bn_to_hash_fc', True):
        hash_fc = nn.Sequential(
            nn.Linear(in_features, nbit, bias=bias),
            nn.BatchNorm1d(nbit)
        )
    elif output_choice == 'zbn':  # kwargs.get('zero_mean_bn', False)
        hash_fc = nn.Sequential(
            nn.Linear(in_features, nbit, bias=bias),
            MeanOnlyBatchNorm(nbit, dim=2)
        )
    elif output_choice == 'bihalf':  # elif kwargs.get('bihalf_to_hash_fc', False):
        hash_fc = nn.Sequential(
            nn.Linear(in_features, nbit, bias=bias),
            BiHalfLayer(kwargs.get('bihalf_gamma', 6))
        )
    else:  # other
        hash_fc = nn.Sequential(
            nn.Linear(in_features, nbit, bias=bias),
            get_hash_activation(output_choice)
        )
    return hash_fc


def get_hash_activation(name='identity'):
    if name == 'identity':
        return nn.Identity()
    elif name == 'signhash':
        return SignHashLayer()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'stochasticbin':
        return StochasticBinaryLayer()
    else:
        return ValueError(f'{name} is not a valid hash activation.')


class Lambda(nn.Module):
    def __init__(self, lambda_func):
        super(Lambda, self).__init__()

        self.lambda_func = lambda_func

    def forward(self, x):
        return self.lambda_func(x)
