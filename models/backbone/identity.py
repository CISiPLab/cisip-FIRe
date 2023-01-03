from models.backbone.base import BaseNet


class Identity(BaseNet):
    def __init__(self, features_size, **kwargs):
        super(Identity, self).__init__()

        self.features_size = features_size

    def forward(self, x):
        return x

    def classify(self, x):
        raise NotImplementedError('no `classify` function for Identity')
