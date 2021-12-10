########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_list():
    return loss_list.keys()


def get_loss_fn(args):
    return loss_list[args.loss]
    return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, conf):
        outputs = outputs[:,:1,:,:]
        return F.l1_loss(outputs, target)


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, conf):
        outputs = outputs[:,:1,:,:]
        return F.mse_loss(outputs, target)


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, conf):
        outputs = outputs[:,:1,:,:]
        return F.smooth_l1_loss(outputs, target)


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, conf):
        outputs = outputs[:,:1,:,:]

        val_pixels = torch.ne(target, 0).float().detach()
        depth_loss = F.l1_loss(outputs*val_pixels, target*val_pixels)
        # print('conf is',conf)
        # print('conf max is ',conf.max())
        # print('conf min is ',conf.min())
        # print('conf mean is ',conf.mean())           
        output_valid = outputs*val_pixels
        target_valid = target*val_pixels
        # print('output_valid is ',output_valid)
        # print('target_valid is ',target_valid)
        # print('output_valid shape is ',output_valid.shape)
        # print('target_valid shape is ',target_valid.shape)
        # print('target_valid max is ',target_valid.max())
        # print('target_valid min is ',target_valid.min())
        # print('target_valid mean is ',target_valid.mean())
        # print('output_valid max is ',output_valid.max())
        # print('output_valid min is ',output_valid.min())
        # print('output_valid mean is ',output_valid.mean())        
        conf_groundtruth = torch.exp( - torch.abs(output_valid - target_valid))
        # print('conf_groundtruth max is ',conf_groundtruth.max())
        # print('conf_groundtruth min is ',conf_groundtruth.min())
        # print('conf_groundtruth mean is ',conf_groundtruth.mean())  
        conf_loss = F.l1_loss(conf*val_pixels, conf_groundtruth)
        print('depth_loss is',depth_loss) 
        print('conf_loss is',conf_loss) 
        loss = depth_loss + conf_loss
        return loss


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, conf):
        outputs = outputs[:,:1,:,:]
        val_pixels = torch.ne(target, 0).float().detach()
        loss = F.mse_loss(outputs*val_pixels, target*val_pixels)
        return loss


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, conf):
        outputs = outputs[:,:1,:,:]
        val_pixels = torch.ne(target, 0).float().detach()
        loss = torch.mean(F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none'))
        return loss


# The proposed probabilistic loss for pNCNN
class MaskedProbLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, targets):
        means = out[:, :1, :, :]
        cout = out[:, 1:2, :, :]

        res = cout
        regl = torch.log(cout+1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss


class MaskedProbExpLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, targets):
        means = out[:, :1, :, :]
        cout = out[:, 1:2, :, :]

        res = torch.exp(cout)  # Residual term
        regl = torch.log(cout+1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss

loss_list = {
    'l1': L1Loss(),
    'l2': L2Loss(),
    'masked_l1': MaskedL1Loss(),
    'masked_l2': MaskedL2Loss(),
    'masked_prob_loss': MaskedProbLoss(),
    'masked_prob_exp_loss': MaskedProbExpLoss(),
}
