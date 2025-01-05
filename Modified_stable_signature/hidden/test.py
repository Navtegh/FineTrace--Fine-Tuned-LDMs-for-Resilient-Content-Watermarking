import utils_model
import torch
import torchvision
from watermarkring import *
from torchvision.utils import save_image
import torch_dct as dct
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

msg_decoder = utils_model.get_hidden_decoder(num_bits=32, redundancy=1, num_blocks=8, channels=64).to(device)
ckpt = utils_model.get_hidden_decoder_ckpt('/w/284/navtegh/stable_signature-main/hidden/output/checkpoint010.pth')
msg_decoder.load_state_dict(ckpt, strict=False)
msg_decoder.eval()
img=torchvision.io.read_image('/w/284/navtegh/stable_signature-main/hidden/delta_w.png')
abc=img.repeat(1,1,1,1)
abc.shape
# b=get_watermarking_pattern(abc/255,15,device)
# c=dct.dct_2d(b)

x=msg_decoder(abc.type(torch.cuda.FloatTensor))
save_image(x[0,:,:,:],'monkey_decoded.png')
# y=msg_decoder(b.type(torch.cuda.FloatTensor))
# y=(y*2)-1
# save_image(y[0,:,:,:],'monkeyW_decoded.png')