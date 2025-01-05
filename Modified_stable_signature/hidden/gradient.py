import torch
import torchvision.transforms.functional as TF
def generate_radial_gradient_image(h):
    # Create a grid of (x, y) coordinates normalized between -1 and 1
    height=256
    y = torch.linspace(1, -1, steps=height).unsqueeze(1).expand(height, height)  # Vertical grid
    x = torch.linspace(1, -1, steps=height).unsqueeze(0).expand(height, height)   # Horizontal grid

    # Compute the radial distance from the center
    radius = torch.sqrt(x**2 + y**2)  # Euclidean distance
    
    # Normalize the distance values to [0, 1]
    max_radius = torch.sqrt(torch.tensor(2.0))  # Max radius in a square [-1, 1] x [-1, 1]
    gradient = radius / max_radius
    # Clip values to ensure they stay between 0 and 1
    gradient = torch.clamp(gradient, 0.0, 1.0)
    
    target_size = h  # Final size for height and width
    current_size = height  # Original size of the image

    # Compute cropping boundaries
    crop_margin = (current_size - target_size) // 2  # Amount to crop from each side


    cropped_tensor = TF.crop(gradient, 
                             top=crop_margin, 
                             left=crop_margin, 
                             height=target_size, 
                             width=target_size)
    return cropped_tensor
