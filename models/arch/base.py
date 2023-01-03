import torch.nn as nn


class BaseNet(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 nbit: int,
                 nclass: int,
                 **kwargs):
        super().__init__()

        self.backbone = backbone
        self.nbit = nbit
        self.nclass = nclass

    def count_parameters(self, mode='trainable'):
        if mode == 'trainable':
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        elif mode == 'non-trainable':
            return sum(p.numel() for p in self.parameters() if not p.requires_grad)
        else:  # all
            return sum(p.numel() for p in self.parameters())

    def finetune_reset(self, *args, **kwargs):
        pass

    def get_backbone(self):
        return self.backbone

    def get_training_modules(self):
        return nn.ModuleDict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError('please implement `forward`')
