# Modified Stable Signature: Rooting Watermarks in Latent Diffusion Models
Group-16, Stuti Wadhwa, Navtegh Singh Gill
## Setup


### Requirements
To install the main dependencies, we recommand using conda.
[PyTorch](https://pytorch.org/) can be installed with:
```cmd
conda install -c pytorch torchvision pytorch==1.12.0 cudatoolkit==11.3
```

Install the remaining dependencies with pip:
```cmd
pip install -r requirements.txt
```

This codebase has been developed with python version 3.8, PyTorch version 1.12.0, CUDA 11.3.


### Models and data

#### Data

The paper uses the [COCO](https://cocodataset.org/) dataset to fine-tune the LDM decoder (we filtered images containing people).
All you need is around 500 images for training (preferably over 256x256).

Code to train the watermark models is available in the folder called `hidden/`.


#### Stable Diffusion models

Create LDM configs and checkpoints from the [Hugging Face](https://huggingface.co/stabilityai) and [Stable Diffusion](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion) repositories.
The code should also work for Stable Diffusion v1 without any change. 
For other models (like old LDMs or VQGANs), you may need to adapt the code to load the checkpoints.

#### Perceptual Losses

The perceptual losses are based on [this repo](https://github.com/SteffenCzolbe/PerceptualSimilarity/).
You should download the weights here: https://github.com/SteffenCzolbe/PerceptualSimilarity/tree/master/src/loss/weights, and put them in a folder called `losses` (this is used in [src/loss/loss_provider.py#L22](https://github.com/facebookresearch/stable_signature/blob/main/src/loss/loss_provider.py#L22)).
To do so you can run 
```
git clone https://github.com/SteffenCzolbe/PerceptualSimilarity.git
cp -r PerceptualSimilarity/src/loss/weights src/loss/losses/
rm -r PerceptualSimilarity
```

## Usage

### Watermark pre-training

Please see [hidden/README.md]for details on how to train the watermark encoder/extractor.
To use different methods files from replacements folder can be used

### Fine-tune LDM decoder

```
python finetune_ldm_decoder.py --num_keys 1 \
    --ldm_config path/to/ldm/config.yaml \
    --ldm_ckpt path/to/ldm/ckpt.pth \
    --msg_decoder_path path/to/msg/decoder/ckpt.torchscript.pt \
    --train_dir path/to/train/dir \
    --val_dir path/to/val/dir
```

This code should generate: 
-  Results of the Finetuning
- `keys.txt`: text file containing the keys used for fine-tuning (one key per line),
- `imgs`: folder containing examples of auto-encoded images.


