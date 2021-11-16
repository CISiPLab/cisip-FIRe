import torch
import torch.nn as nn

from models import register_network, BaseArch
from models.architectures.helper import get_backbone, get_hash_activation


def build_adjacency_hamming(tensor_in):
    """
    Hamming-distance-based graph. It is self-connected.
    :param tensor_in: [N D]
    :return:
    """
    code_length = tensor_in.size(1)
    m1 = tensor_in - 1.
    c1 = torch.matmul(tensor_in, m1.t())  # (N, N)
    c2 = torch.matmul(m1, tensor_in.t())  # (N, N)
    normalized_dist = torch.abs(c1 + c2) / code_length
    return torch.pow(1 - normalized_dist, 1.4)  # why 1.4?


class TwinBottleneck(nn.Module):
    def __init__(self, bbn_dim, cbn_dim, **kwargs):
        super().__init__()
        self.bbn_dim = bbn_dim
        self.cbn_dim = cbn_dim
        self.gcn = GCNLayer(cbn_dim, cbn_dim)

    def forward(self, bbn, cbn):
        adj = build_adjacency_hamming(bbn)
        return torch.sigmoid(self.gcn(cbn, adj))


class GCNLayer(nn.Module):
    # https://github.com/ymcidence/TBH/blob/778dd1cfb5c631d109493e0cee858ab6fa675416/layer/gcn.py#L8

    def __init__(self, in_dim=512, out_dim=512, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, values, adjacency, **kwargs):
        """
        :param values:
        :param adjacency:
        :param kwargs:
        :return:
        """
        return self.spectrum_conv(values, adjacency)

    def spectrum_conv(self, values, adjacency):
        """
        Convolution on a graph with graph Laplacian
        :param values: [N D]
        :param adjacency: [N N] must be self-connected
        :return:
        """
        fc_sc = self.fc(values)  # (N, D)
        conv_sc = self.graph_laplacian(adjacency) @ fc_sc  # (N, D)
        return conv_sc

    def graph_laplacian(self, adjacency):
        """
        :param adjacency: must be self-connected
        :return:
        """
        graph_size = adjacency.size(0)  # (BS, BS)
        d = adjacency @ torch.ones(size=[graph_size, 1]).to(adjacency.device)  # (BS, 1)
        d_inv_sqrt = torch.pow(d + 1e-8, -0.5)  # (BS, 1)
        d_inv_sqrt = torch.eye(graph_size).to(adjacency.device) * d_inv_sqrt  # (BS, BS)
        laplacian = d_inv_sqrt @ adjacency @ d_inv_sqrt  # (BS, BS)
        return laplacian


class Encoder(nn.Module):

    def __init__(self, input_dim=4096, middle_dim=1024, bbn_dim=64, cbn_dim=512):
        """
        :param middle_dim: hidden units
        :param bbn_dim: binary bottleneck size
        :param cbn_dim: continuous bottleneck size
        """
        super(Encoder, self).__init__()
        self.code_length = bbn_dim
        self.fc_1 = nn.Sequential(
            nn.Linear(input_dim, middle_dim),
            nn.ReLU()
        )
        self.fc_2_1 = nn.Sequential(
            nn.Linear(middle_dim, bbn_dim)
        )

        self.fc_2_2 = nn.Sequential(
            nn.Linear(middle_dim, cbn_dim),
            # nn.Sigmoid()
        )  # paper is Identity
        self.hash_layer = get_hash_activation('stochasticbin')

    def forward(self, x):
        fc_1 = self.fc_1(x)
        bbn = self.fc_2_1(fc_1)
        if self.training:
            bbn, _ = self.hash_layer(bbn, torch.ones_like(bbn) * 0.5)
        else:  # eval mode, just output sigmoid probability
            bbn = torch.sigmoid(bbn)
        cbn = self.fc_2_2(fc_1)
        return bbn, cbn


class Decoder(nn.Module):
    def __init__(self, in_dim, middle_dim, feat_dim):
        """
        :param middle_dim: hidden units
        :param feat_dim: data dim
        """
        super(Decoder, self).__init__()
        self.fc_1 = nn.Sequential(
            nn.Linear(in_dim, middle_dim),
            nn.ReLU()
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(middle_dim, feat_dim)
        )  # original implementation is with ReLU, but paper is Identity

    def forward(self, x):
        fc_1 = self.fc_1(x)
        return self.fc_2(fc_1)


@register_network('tbh')
class ArchTBH(BaseArch):
    """Arch TBH"""

    def __init__(self, config, **kwargs):
        super(ArchTBH, self).__init__(config, **kwargs)
        self.backbone = get_backbone(backbone=self.backbone_name, nbit=self.nbit, nclass=self.nclass,
                                     pretrained=self.pretrained, freeze_weight=self.freeze_weight, **kwargs)

        self.encoder = Encoder(self.backbone.in_features, 1024, self.nbit, 512)
        self.decoder = Decoder(512, 1024, self.backbone.in_features)
        self.tbh = TwinBottleneck(self.nbit, 512)

        self.discriminator_binary = nn.Sequential(
            # nn.Linear(nbit, 1024),
            # nn.ReLU(),
            nn.Linear(self.nbit, 1),
            nn.Sigmoid()
        )  # original implementation is one FC-sigmoid layer. paper is one FC-ReLU layer + one FC-sigmoid layer
        self.discriminator_continuous = nn.Sequential(
            # nn.Linear(512, 1024),
            # nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )  # original implementation is one FC-sigmoid layer. paper is one FC-ReLU layer + one FC-sigmoid layer

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return (list(self.encoder.parameters()) +
                list(self.decoder.parameters()) +
                list(self.tbh.parameters()))

    def get_discriminator_params(self):
        return (list(self.discriminator_binary.parameters()) +
                list(self.discriminator_continuous.parameters()))

    def forward(self, x):
        x = self.backbone(x)

        bbn, cbn = self.encoder(x)
        gcn_cbn = self.tbh(bbn, cbn)
        rec_x = self.decoder(gcn_cbn)
        dis1_real = self.discriminator_binary(bbn)
        dis1_fake = self.discriminator_binary(torch.bernoulli(torch.ones_like(bbn) * 0.5))
        dis2_real = self.discriminator_continuous(gcn_cbn)
        dis2_fake = self.discriminator_continuous(torch.rand_like(gcn_cbn))

        return x, bbn, rec_x, [(dis1_real, dis1_fake), (dis2_real, dis2_fake)]


if __name__ == '__main__':
    torch.manual_seed(1234)
    net = ArchTBH(64, 10, False, False, 'alexnet')
    print(net.training)
    net.train()

    data = torch.randn(1, 3, 224, 224)
    x, bbn, rec_x, discs = net(data)
    print(x)
    print(bbn)
    print(rec_x)
    print(discs)

    from functions.loss.tbh import TBHLoss

    criterion = TBHLoss()

    opt_A = torch.optim.Adam([{'params': net.get_hash_params()},
                              {'params': net.get_features_params()}],
                             0.0001)
    hashparams = net.get_hash_params()
    opt_C = torch.optim.Adam([{'params': net.get_discriminator_params()}],
                             0.0001)
    disparams = net.get_discriminator_params()

    loss = criterion(x, bbn, rec_x, discs)
    print(loss)

    actor_loss = criterion.losses['actor']
    critic_loss = criterion.losses['critic']
    #
    params = [p for param in opt_A.param_groups for p in param['params']]
    actor_loss.backward(retain_graph=True, inputs=params)
    opt_A.step()  # step for main hashing flow

    params = [p for param in opt_C.param_groups for p in param['params']]
    critic_loss.backward(inputs=params)
    opt_C.step()
