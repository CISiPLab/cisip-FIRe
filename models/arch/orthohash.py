import torch
import torch.nn as nn
from transformers.models.clip import CLIPModel
from transformers.models.vit import ViTModel

from models.arch.base import BaseNet
from models.layers.adapter import ViTLayerWithAdapter, CLIPEncoderLayerWithAdapter
from models.layers.cossim import CosSim
from models.layers.lambda_layer import Lambda


class OrthoHash(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 codebook: torch.Tensor,
                 **kwargs):
        super().__init__(backbone, nbit, nclass, **kwargs)

        self.register_buffer('codebook', codebook)

        if self.codebook is None:  # usual CE
            self.ce_fc = nn.Linear(self.nbit, self.nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(self.nbit, self.nclass, self.codebook, learn_cent=False)

        self.hash_fc = nn.Sequential(
            nn.Linear(self.backbone.features_size, self.nbit, bias=False),
            nn.BatchNorm1d(self.nbit, momentum=0.1)
        )

    def finetune_reset(self, nclass, codebook):
        self.nclass = nclass
        self.codebook = codebook
        if self.codebook is None:  # usual CE
            self.ce_fc = nn.Linear(self.nbit, nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(self.nbit, nclass, self.codebook, learn_cent=False)

    def get_training_modules(self):
        return nn.ModuleDict({'ce_fc': self.ce_fc, 'hash_fc': self.hash_fc})

    def forward(self, x):
        x = self.backbone(x)
        v = self.hash_fc(x)
        u = self.ce_fc(v)
        return u, v


class OrthoHashWithAdapter(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 codebook: torch.Tensor,
                 adapter_bottleneck_dim: float = 512,
                 **kwargs):
        super().__init__(backbone, nbit, nclass)

        self.register_buffer('codebook', codebook)

        self.trainable_params = nn.ParameterDict()

        if isinstance(self.backbone.model, ViTModel):
            self.backbone = self.backbone.model  # type: ViTModel

            vision_model = self.backbone.encoder
            current_layers = vision_model.layer

            new_layers = nn.ModuleList([ViTLayerWithAdapter(vision_model.config) for _ in current_layers])
            for i, (nlayer, clayer) in enumerate(zip(new_layers, current_layers)):  # type: ViTLayerWithAdapter
                nlayer.load_state_dict(clayer.state_dict())
                nlayer.setup_adapt_mlp(adapter_bottleneck_dim, 0.0)
                nlayer_named_params = nlayer.get_adapt_params()
                for pname, param in nlayer_named_params:
                    self.trainable_params[f'adapter_{i}_{pname.replace(".", "_")}'] = param
            vision_model.layer = new_layers
            hidden_size = vision_model.config.hidden_size

            self.backbone.pooler = Lambda(lambda x: x[:, 0])
        elif isinstance(self.backbone.model, CLIPModel):
            self.backbone = self.backbone.model.vision_model
            vision_model = self.backbone
            current_layers = vision_model.encoder.layers

            clip_encoder_cfg = vision_model.encoder.config
            new_layers = nn.ModuleList([CLIPEncoderLayerWithAdapter(clip_encoder_cfg) for _ in current_layers])
            for i, (nlayer, clayer) in enumerate(zip(new_layers, current_layers)):  # type: CLIPEncoderLayerWithAdapter
                nlayer.load_state_dict(clayer.state_dict())
                nlayer.setup_adapt_mlp(adapter_bottleneck_dim, 0.0)
                nlayer_named_params = nlayer.get_adapt_params()
                for pname, param in nlayer_named_params:
                    self.trainable_params[f'adapter_{i}_{pname.replace(".", "_")}'] = param
            vision_model.encoder.layers = new_layers
            hidden_size = vision_model.config.hidden_size
        else:
            raise ValueError

        self.backbone.requires_grad_(False)

        # self.trainable_params['logit_scale'] = self.backbone.logit_scale
        for param_key in self.trainable_params:
            self.trainable_params[param_key].requires_grad_(True)

        self.hash_fc = nn.Sequential(
            nn.Linear(hidden_size, self.nbit, bias=False),
            nn.BatchNorm1d(self.nbit, momentum=0.1)
        )

        if self.codebook is None:  # usual CE
            self.ce_fc = nn.Linear(self.nbit, self.nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(self.nbit, self.nclass, self.codebook, learn_cent=False)

    def get_backbone(self):
        return nn.Identity()

    def get_training_modules(self):
        return nn.ModuleDict({'trainable_params': self.trainable_params,
                              'ce_fc': self.ce_fc,
                              'hash_fc': self.hash_fc})

    def forward(self, x):
        x = self.backbone(x).pooler_output
        x = self.hash_fc(x)
        u = self.ce_fc(x)
        return u, x
