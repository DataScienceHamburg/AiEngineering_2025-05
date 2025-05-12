#%%
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
# %%
img_path = "kiki.jpg"
img = Image.open(img_path)
img

#%% preprocessing steps
preprocess_steps = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.TenCrop((224, 224))
    # transforms.RandomRotation(90),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    # transforms.RandomRotation(30)
    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

x = preprocess_steps(img)
x
# %%
