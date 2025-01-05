import torch
import torch_dct as dct
from torchvision.utils import save_image
import copy
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def circle_mask(size=256, r=100, x_offset=110, y_offset=-110):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0)**2 + (y-y0)**2)<= r**2

def get_watermarking_pattern( image,w_radius, device, it):
    gt_init = (image).to(device)
    gt_patch=dct.dct_2d(gt_init)
#     gt_patch_tmp = copy.deepcopy(gt_patch)
    for i in range(w_radius, 0, -1):
        tmp_mask = circle_mask(gt_init.shape[-1], r=i)
        tmp_mask = torch.tensor(tmp_mask).to(device)
        for j in range(gt_patch.shape[1]):
            if it%2==0:
                if i % 6 == 0 or i % 6 == 1 or i % 6 == 2:  
                    gt_patch[:, j, tmp_mask] = torch.tensor(1, dtype=torch.uint8)
                else: 
                    gt_patch[:, j, tmp_mask] = torch.tensor(0, dtype=torch.uint8)
            else:
                if i % 6 == 0 or i % 6 == 1 or i % 6 == 2:  
                    gt_patch[:, j, tmp_mask] = torch.tensor(0, dtype=torch.uint8)
                else: 
                    gt_patch[:, j, tmp_mask] = torch.tensor(1, dtype=torch.uint8)
    # save_image(gt_patch[0,:,:,:],"gt_patch.png")
    xya=dct.idct_2d(gt_patch)
#     save_image(xya[0,:,:,:],"xyz.png")
    
    return xya

def get_watermarking_mask(image,w_radius, device):
    gt_init = (image).to(device)
    gt_patch=gt_init
    gt_init=gt_init*0
    tmp_mask = circle_mask(gt_init.shape[-1], r=w_radius)
    tmp_mask = torch.tensor(tmp_mask).to(device)
    gt_init[:,tmp_mask] = gt_patch[:,tmp_mask]
    gt_init=gt_init[:,-33:,-32:]
    gt_init=gt_init[:,:-4,:-3]
    return gt_init

def get_key(batch,size,w_radius, device):
    gt_init =  torch.zeros((batch, size, size), dtype=torch.uint8)
    gt_patch = gt_init
    for i in range(w_radius, 0, -1):
        tmp_mask = circle_mask(gt_init.shape[-1], r=i)
        tmp_mask = torch.tensor(tmp_mask).to(device)
        if i % 6 == 0 or i % 6 == 1 or i % 6 == 2:  
            gt_patch[:, tmp_mask] = torch.tensor(1, dtype=torch.uint8)
        else: 
            gt_patch[:, tmp_mask] = torch.tensor(0, dtype=torch.uint8)
    gt_patch=gt_patch[:,-33:,-32:]
    gt_patch=gt_patch[:,:-4,:-3]
    return gt_patch