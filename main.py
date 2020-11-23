import os
import torch
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

from models.unet import UNet
from models.skip import skip
from models.resnet import ResNet
from utils.inpainting_utils import *

dtype = torch.FloatTensor
PLOT = True
imsize = -1
dim_div_by = 64

## Figure
img_path  = 'data/original.png'
mask_path = 'data/mask.png'
out_path = 'data/output.png'

NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet

# Load mask
img_pil, img_np = get_image(img_path, imsize)
img_mask_pil, img_mask_np = get_image(mask_path, imsize)

# Center crop
img_mask_pil = crop_image(img_mask_pil, dim_div_by)
img_pil = crop_image(img_pil, dim_div_by)

img_np = pil_to_np(img_pil)
img_mask_np = pil_to_np(img_mask_pil)

# Visualize
# img_mask_var = np_to_torch(img_mask_np).type(torch.FloatTensor)
# plot_image_grid([img_np, img_mask_np, img_mask_np*img_np], 3,11)

# Setup
pad = 'reflection'
OPT_OVER = 'net'
OPTIMIZER = 'adam'
INPUT = 'noise'
input_depth = 1
iteration = 500
figsize = 8
reg_noise_std = 0.00
param_noise = True

if 'skip' in NET_TYPE:
    LR = 0.01
    depth = int(NET_TYPE[-1])
    net = skip(input_depth, img_np.shape[0],
            num_channels_down = [16, 32, 64, 128, 128, 128][:depth],
            num_channels_up =   [16, 32, 64, 128, 128, 128][:depth],
            num_channels_skip = [0, 0, 0, 0, 0, 0][:depth],
            filter_size_up=3, filter_size_down=5, filter_skip_size=1,
            upsample_mode='nearest', need1x1_up=False, need_sigmoid=True,
            need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
elif NET_TYPE == 'UNET':
    LR = 0.001
    net = UNet(num_input_channels=input_depth, num_output_channels=3,
                feature_scale=8, more_layers=1,
                concat_x=False, upsample_mode='deconv',
                pad='zero', norm_layer=torch.nn.InstanceNorm2d, need_sigmoid=True, need_bias=True)
    param_noise = False
elif NET_TYPE == 'ResNet':
    LR = 0.001
    net = ResNet(input_depth, img_np.shape[0], 8, 32, need_sigmoid=True, act_fun='LeakyReLU')
    param_noise = False

else:
    assert False

net = net.type(dtype)
net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

# Compute number of parameters
s  = sum(np.prod(list(p.size())) for p in net.parameters())
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)

# Main loop
i = 0
def closure():
    global i
    if param_noise:
        for n in [x for x in net.parameters() if len(x.size()) == 4]:
            n = n + n.detach().clone().normal_() * n.std() / 50
    
    net_input = net_input_saved
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)
    total_loss = mse(out * mask_var, img_var * mask_var)
    total_loss.backward()

    print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
    i += 1
    return total_loss

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, iteration)

out_np = torch_to_np(net(net_input))
out_img = np_to_pil(out_np)
plot_image_grid([out_np], factor=5)
save_image(out_path, out_img)