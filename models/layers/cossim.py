import torch
import torch.nn as nn
import torch.nn.functional as F


class CosSim(nn.Module):
    def __init__(self,
                 nfeat,
                 nclass,
                 codebook=None,
                 learn_cent=True,
                 signhash=None,
                 learn_cent_assign=False,
                 group=1):
        super(CosSim, self).__init__()
        self.nfeat = nfeat
        self.nclass = nclass
        self.learn_cent = learn_cent
        self.learn_cent_assign = learn_cent_assign
        self.group = group

        if codebook is None:  # if no centroids, by default just usual weight
            codebook = torch.randn(nclass, nfeat)

        if not learn_cent:
            self.register_buffer('centroids', codebook.clone())
        else:
            if isinstance(codebook, nn.Parameter):
                self.centroids = codebook
            else:
                self.centroids = nn.Parameter(codebook.clone())

        if self.learn_cent_assign:
            self.cent_assign = nn.Parameter(torch.eye(self.nclass))
        else:
            self.register_buffer('cent_assign', torch.eye(self.nclass))

        self.signhash = signhash

    def forward(self, x):
        if self.signhash is not None:
            centroids = self.signhash(self.centroids)
        else:
            centroids = self.centroids

        # cent_assign = self.cent_assign.clamp(min=0)
        # # cent_assign = cent_assign / cent_assign.sum(dim=1, keepdim=True)
        #
        # centroids = cent_assign @ centroids
        centroids_group = centroids.reshape(self.nclass, self.group, -1)
        x_group = x.reshape(x.size(0), self.group, -1)

        nfeat = F.normalize(x_group, p=2, dim=-1)
        ncenters = F.normalize(centroids_group, p=2, dim=-1)

        nfeat = nfeat.reshape(x.size(0), -1)
        ncenters = ncenters.reshape(self.nclass, -1)

        logits = torch.matmul(nfeat, ncenters.t()) / self.group

        return logits

    def extra_repr(self) -> str:
        return 'in_features={}, n_class={}, learn_cent={}, learn_cent_assign={}'.format(
            self.nfeat, self.nclass, self.learn_cent, self.learn_cent_assign
        )
