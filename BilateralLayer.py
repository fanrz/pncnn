import torch
import argparse
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import sys

import BilateralGrid as bs


##############################################################################
REQUIRES_CONF_GRAD = True
##############################################################################


# depthBs(input, out)
def BilateralFunction(image, pred, grid_params_arr, bs_params_arr ):
    batch_size, channel_num, height, width = pred.size()
    output = np.zeros((batch_size, height, width, channel_num), np.float32)
    confidence = np.ones((batch_size, 1, height, width))

    image_np = image.cpu().numpy().swapaxes(1, 2).swapaxes(2, 3)
    pred_np = pred.numpy().swapaxes(1, 2).swapaxes(2, 3)
    conf_np = confidence.squeeze(1)

    grid_params = {}
    grid_params['sigma_luma'] = grid_params_arr[0].data.item()
    grid_params['sigma_chroma'] = grid_params_arr[1].data.item()
    grid_params['sigma_spatial'] = grid_params_arr[2].data.item()

    bs_params = {}
    bs_params['lam'] =  bs_params_arr[0].data.item()
    bs_params['A_diag_min'] = bs_params_arr[1].data.item()
    bs_params['cg_tol'] = bs_params_arr[2].data.item()
    bs_params['cg_maxiter'] = bs_params_arr[2].data.item()

    for i in range(batch_size):
        curr_image = image_np[i, :, :, :]
        curr_pred = pred_np[i, :, :, :]
        curr_conf = conf_np[i, :, :]
        im_shape = curr_pred.shape

        grid = bs.BilateralGrid(curr_image*255.0, **grid_params)

        curr_result, _ = bs.solve(grid, curr_pred, curr_conf, bs_params, im_shape)
        output[i, :, :, :] = curr_result

    output = output.swapaxes(3, 2).swapaxes(2, 1)
    return output


class BilateralLayer(object):
    def __init__(self):
        super(BilateralLayer, self).__init__()
        # bilateral solver for normal
        self.grid_params = {
            'sigma_luma' : 4, #Brightness bandwidth
            'sigma_chroma': 2, # Color bandwidth
            'sigma_spatial': 4# Spatial bandwidth
        }

        self.bs_params = {
            'lam': 100, # The strength of the smoothness parameter
            'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
            'cg_tol': 1e-5, # The tolerance on the convergence in PCG
            'cg_maxiter': 10 # The number of PCG iterations
        }

        self.grid_params_arr = Variable(torch.FloatTensor(3) )
        self.bs_params_arr = Variable(torch.FloatTensor(4) )

        self.grid_params_arr[0] = self.grid_params['sigma_luma']
        self.grid_params_arr[1] = self.grid_params['sigma_chroma']
        self.grid_params_arr[2] = self.grid_params['sigma_spatial']

        self.bs_params_arr[0] = self.bs_params['lam']
        self.bs_params_arr[1] = self.bs_params['A_diag_min']
        self.bs_params_arr[2] = self.bs_params['cg_tol']
        self.bs_params_arr[3] = self.bs_params['cg_maxiter']

    def solve(self, input, pred):
        # outBsPred = depthBs(input, out)
        image = input[:,0:3,:,:]

        # def forward(image, pred, grid_params_arr, bs_params_arr ):

        out = BilateralFunction(image, pred, self.grid_params_arr, self.bs_params_arr )
        return out

