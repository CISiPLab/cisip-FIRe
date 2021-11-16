import torch
import torch.nn as nn
import torch.nn.functional as F


def adv_loss(real, fake):
    real_loss = torch.mean(F.binary_cross_entropy(real, torch.ones_like(real)))
    fake_loss = torch.mean(F.binary_cross_entropy(fake, torch.zeros_like(fake)))
    total_loss = real_loss + fake_loss
    return total_loss


class TBHLoss(nn.Module):
    def __init__(self, adv=1, rec_method='element', **kwargs):
        super().__init__()

        self.adv = adv
        self.rec_method = rec_method

        self.losses = {}

    def forward(self, x, code_logits, rec_x, discs):
        rec_loss = torch.sum((rec_x - x).pow(2)) * 0.5  # tf.nn.l2_loss is sum(x ** 2) / 2
        divisor = 1
        if self.rec_method == 'batch':
            divisor = x.size(0)
        elif self.rec_method == 'all':  # including bits
            divisor = x.size(0) * x.size(1)
        rec_loss = rec_loss / divisor

        dis1_real, dis1_fake = discs[0]
        dis2_real, dis2_fake = discs[1]

        # min real max fake, no backprop into discriminator
        actor_loss = rec_loss - self.adv * (adv_loss(dis1_real, dis1_fake) + adv_loss(dis2_real, dis2_fake))

        # min fake max real, only backprop into discriminator
        critic_loss = self.adv * (adv_loss(dis1_real, dis1_fake) + adv_loss(dis2_real, dis2_fake))

        total_loss = actor_loss + critic_loss

        self.losses['rec'] = rec_loss
        self.losses['real'] = actor_loss - rec_loss
        self.losses['actor'] = actor_loss
        self.losses['critic'] = critic_loss

        return total_loss
