import torch
import torch_dct as dct
from torchvision.utils import save_image
import copy
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def circle_mask(size=256, r=100, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0)**2 + (y-y0)**2)<= r**2

def get_watermarking_pattern( batch_size,num_channels_latents , height, width,w_radius,w_channel, device):
    gt_init = torch.ones((batch_size,num_channels_latents , height, width)).to(device)
    gt_patch=gt_init
    gt_patch_tmp = copy.deepcopy(gt_patch)
    gt_patch=gt_patch*0
    for i in range(w_radius, 0, -1):
        tmp_mask = circle_mask(gt_init.shape[-1], r=i)
        tmp_mask = torch.tensor(tmp_mask).to(device)            
        for j in range(gt_patch.shape[1]):
            if i % 6 == 0 or i % 6 == 1 or i % 6 == 2:  
                gt_patch[:, j, tmp_mask] = 1
            else: 
                gt_patch[:, j, tmp_mask] = 0
                
    init_latents_w=torch.ones((batch_size,num_channels_latents, height, width)).to(device)
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)
    np_mask = circle_mask(init_latents_w.shape[-1], r=w_radius)
    torch_mask = torch.tensor(np_mask).to(device)
    watermarking_mask[:,w_channel] = torch_mask
    init_latents_w_fft = dct.dct_2d(init_latents_w)
    init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    init_latents_w = dct.idct_2d(init_latents_w_fft)  
#     xyz=dct.dct_2d(init_latents_w[0,3,:,:])
#     save_image(init_latents_w[0,3,:,:],"whatwewant.png")
#     save_image(gt_patch[0,0,:,:],"gt_patch.png")
#     save_image(xyz.real,"xyz_image.png")
    return init_latents_w[0,3,:,:], gt_patch[0,0,:,:]