import torch
import torch.nn as nn


class CSQLoss(nn.Module):
    """
    https://github.com/swuxyj/DeepHash-pytorch/blob/master/CSQ.py

    Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020

    similar to DPN but with BCELoss
    """

    def __init__(self, nbit, codebook: torch.Tensor, lambda_q=0.001, multiclass=False, **kwargs):
        super(CSQLoss, self).__init__()
        self.multiclass = multiclass
        self.lambda_q = lambda_q
        self.criterion = nn.BCELoss()
        self.multi_label_random_center = torch.randint(2, (nbit,)).float()
        self.losses = {}

        self.codebook = codebook

    def forward(self, code_logits, labels, onehot=True):
        assert self.codebook is not None
        code_logits = code_logits.tanh()
        hash_center = self.label2center(labels, onehot)

        loss_c = self.criterion(0.5 * (code_logits + 1), 0.5 * (hash_center + 1))
        loss_q = (code_logits.abs() - 1).pow(2).mean()
        self.losses['center'] = loss_c
        self.losses['quant'] = loss_q

        loss = loss_c + self.lambda_q * loss_q
        return loss

    def label2center(self, y, onehot):
        if not self.multiclass:
            if onehot:
                hash_center = self.codebook[y.argmax(axis=1)]
            else:
                hash_center = self.codebook[y]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y.float() @ self.codebook
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0].to(y.device)
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center
