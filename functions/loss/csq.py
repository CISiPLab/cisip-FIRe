import torch
import torch.nn as nn


class CSQLoss(nn.Module):
    """https://github.com/swuxyj/DeepHash-pytorch/blob/master/CSQ.py
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf
    """
    def __init__(self, multiclass, nbit, device, lambda_q=0.001, scale_c=1., **kwargs):
        super(CSQLoss, self).__init__()
        device = torch.device(device)
        self.multiclass = multiclass
        self.lambda_q = lambda_q
        self.scale_c = scale_c
        self.criterion = nn.BCELoss()
        self.multi_label_random_center = torch.randint(2, (nbit,)).float().to(device)
        self.losses = {}

        self.centroids = None

    def forward(self, logits, code_logits, labels, onehot=True):
        assert self.centroids is not None
        code_logits = code_logits.tanh()
        hash_center = self.label2center(labels, onehot)

        loss_c = self.criterion(0.5 * (code_logits + 1), 0.5 * (hash_center + 1))
        loss_q = (code_logits.abs() - 1).pow(2).mean()
        self.losses['center'] = loss_c
        self.losses['quant'] = loss_q

        loss = self.scale_c * loss_c + self.lambda_q * loss_q
        return loss

    def label2center(self, y, onehot):
        if not self.multiclass:
            if onehot:
                hash_center = self.centroids[y.argmax(axis=1)]
            else:
                hash_center = self.centroids[y]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y.float() @ self.centroids
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center
