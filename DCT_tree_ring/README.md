# Modification to Tree-ring watermarking technique to improve structural similarity
Group-16, Stuti Wadhwa, Navtegh Singh Gill
This repo contains modification of the official implementation of [Tree-Ring Watermarks](http://arxiv.org/abs/2305.20030) for the course project of CSC 2541 Generative AI for Images, Fall 2024 

## Usage
We have implemented discrete cosine tranformation instead of fft in the original work to improve structural similarity. Structural similarity is obtained quantitatively via LPIPS and SSIM losses.  
To run the experiments with fft or dct transformations, change `config.py`, and then use the following commands.
### Perform main experiments and calculate CLIP Score
For non-adversarial case, you can simply run requirements and the below code:
```
python run_tree_ring_watermark.py --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k
```
