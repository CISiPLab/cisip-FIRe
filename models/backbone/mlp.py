import torch.nn as nn

from models.backbone.base_backbone import BaseBackbone


class MLPBackbone(BaseBackbone):
    def __init__(self, nclass, nbit, in_channels=2048,
                 mlp_hidden_channels=None, mlp_out_channels=None,
                 mlp_activation='gelu', mlp_drop_prob=0.0, **kwargs):
        super(MLPBackbone, self).__init__()

        self.in_features = in_channels
        self.nbit = nbit
        self.nclass = nclass

        out_features = in_channels or mlp_out_channels
        hidden_features = in_channels or mlp_hidden_channels
        act_layer = self.get_act_layer(mlp_activation)
        drop_prob = mlp_drop_prob

        self.fc1 = nn.Linear(self.in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_prob)

    def get_act_layer(self, mlp_activation):
        if mlp_activation == 'gelu':
            return nn.GELU
        elif mlp_activation == 'relu':
            return nn.ReLU
        elif mlp_activation == 'leakyrelu':
            return nn.LeakyReLU
        else:
            raise NotImplementedError('activation layer not implemented')

    def get_features_params(self):
        return list(self.parameters())

    def get_hash_params(self):
        raise NotImplementedError('no hash layer in backbone')

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
