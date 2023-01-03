import torch
import torch.nn as nn

from models import register_network
from models.arch.base import BaseArch
from models.layers.cossim import CosSim


@register_network('unsup_orthohash')
class UnsupOrthoHashArch(BaseArch):
    def __init__(self,
                 config,
                 **kwargs):
        super(UnsupOrthoHashArch, self).__init__(config, **kwargs)

        self.codebook = kwargs.get('codebook')
        self.nclass = config['method']['param']['k']

        if self.codebook is None:  # usual CE
            self.ce_fc = nn.Linear(self.nbit, self.nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(self.nbit, self.nclass, self.codebook, learn_cent=False)

        self.hash_fc = nn.Sequential(
            nn.Linear(self.backbone.features_size, self.nbit, bias=False),
            nn.BatchNorm1d(self.nbit, momentum=0.1)
        )
        self.register_buffer('center', torch.zeros(1, self.nclass))

    @torch.no_grad()
    def momentum_update(self, teacher, m):
        """

        :param teacher: a model of UnsupOrthoHashArch
        :param m: momentum
        :return:
        """
        student_params = list(self.backbone.parameters()) + list(self.hash_fc.parameters())
        teacher_params = list(teacher.backbone.parameters()) + list(teacher.hash_fc.parameters())

        for param_s, param_t in zip(student_params, teacher_params):
            param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

        student_bns = [module for module in self.modules() if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))]
        teacher_bns = [module for module in teacher.modules() if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))]

        # direct copy for BN running mean and var
        for bn_s, bn_t in zip(student_bns, teacher_bns):
            bn_t.running_mean = bn_s.running_mean
            bn_t.running_var = bn_s.running_var
            bn_t.num_batches_tracked = bn_s.num_batches_tracked

        teacher.center = self.center

    @torch.no_grad()
    def update_center(self, t_logits, m=0.9):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(t_logits, dim=0, keepdim=True)  # (bs, c) -> (1, c)

        # ema update
        self.center = self.center * m + batch_center * (1 - m)

    def get_training_modules(self):
        return nn.ModuleDict({'ce_fc': self.ce_fc, 'hash_fc': self.hash_fc})

    def forward(self, x):
        x = self.backbone(x)
        v = self.hash_fc(x)
        u = self.ce_fc(v)
        return u, x, v


@register_network('unsup_orthohash_v2')
class UnsupOrthoHashv2Arch(BaseArch):
    def __init__(self,
                 config,
                 **kwargs):
        super(UnsupOrthoHashv2Arch, self).__init__(config, **kwargs)

        self.codebook = kwargs.get('codebook')
        self.nclass = config['method']['param']['k']

        if self.codebook is None:  # usual CE
            self.ce_fc = nn.Linear(self.nbit, self.nclass)
        else:
            # not learning cent, we are doing codebook learning
            self.ce_fc = CosSim(self.nbit, self.nclass, self.codebook, learn_cent=False)

        self.ce_fc_2 = nn.Linear(self.backbone.features_size, self.nclass)

        self.hash_fc = nn.Sequential(
            nn.Linear(self.backbone.features_size, self.nbit, bias=False),
            nn.BatchNorm1d(self.nbit, momentum=0.1)
        )
        self.register_buffer('center', torch.zeros(1, self.nclass))

    @torch.no_grad()
    def momentum_update(self, teacher, m):
        """

        :param teacher: a model of UnsupOrthoHashArch
        :param m: momentum
        :return:
        """
        student_params = list(self.backbone.parameters()) + list(self.hash_fc.parameters())
        teacher_params = list(teacher.backbone.parameters()) + list(teacher.hash_fc.parameters())

        for param_s, param_t in zip(student_params, teacher_params):
            param_t.data.mul_(m).add_((1 - m) * param_s.detach().data)

        student_bns = [module for module in self.modules() if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))]
        teacher_bns = [module for module in teacher.modules() if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))]

        # direct copy for BN running mean and var
        for bn_s, bn_t in zip(student_bns, teacher_bns):
            bn_t.running_mean = bn_s.running_mean
            bn_t.running_var = bn_s.running_var
            bn_t.num_batches_tracked = bn_s.num_batches_tracked

        teacher.center = self.center

    @torch.no_grad()
    def update_center(self, t_logits, m=0.9):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(t_logits, dim=0, keepdim=True)  # (bs, c) -> (1, c)

        # ema update
        self.center = self.center * m + batch_center * (1 - m)

    def get_training_modules(self):
        return nn.ModuleDict({'ce_fc': self.ce_fc, 'hash_fc': self.hash_fc, 'ce_fc_2': self.ce_fc_2})

    def forward(self, x):
        x = self.backbone(x)
        v = self.hash_fc(x)
        u = self.ce_fc(v)
        y = self.ce_fc_2(x)
        return y, u, x, v
