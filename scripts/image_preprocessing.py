#%% packages
import torch
from torchvision import transforms
from PIL import Image

# %%
img = Image.open("AiFuture.png")
img

#%% 
preprocess_steps = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.CenterCrop(500),
    
    transforms.Resize(size=(300, 300)),
    transforms.Grayscale(),
    transforms.ToTensor()
])
preprocess_steps(img).shape
# %%
