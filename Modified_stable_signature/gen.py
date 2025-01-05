import torch 
device = torch.device('cuda')

from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from utils_model import load_model_from_config 

ldm_config = "/w/284/navtegh/stable_signature-main/v2-inference.yaml"
ldm_ckpt = "/w/284/navtegh/stable_signature-main/v2-1_512-ema-pruned.ckpt"

print(f'>>> Building LDM model with config {ldm_config} and weights from {ldm_ckpt}...')
config = OmegaConf.load(f"{ldm_config}")
ldm_ae = load_model_from_config(config, ldm_ckpt)
ldm_aef = ldm_ae.first_stage_model
ldm_aef.eval()

# loading the fine-tuned decoder weights
state_dict = torch.load("/w/284/navtegh/stable_signature-main/output/checkpoint_009.pth")['ldm_decoder']
unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
print(unexpected_keys)
print("you should check that the decoder keys are correctly matched")

# loading the pipeline, and replacing the decode function of the pipe
model = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to(device)
pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))
from torch.cuda.amp import autocast
with autocast(enabled=True):
    img = pipe("the cat drinks water.").images[0]
img.save("cat.png")