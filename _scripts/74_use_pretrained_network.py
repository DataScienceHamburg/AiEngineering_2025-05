#%% packages
import torch
import torchvision
from torchvision import models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import OrderedDict


# %% image transformations
transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize((299, 299)),
    # torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
])

# %%
BATCH_SIZE = 1
test_folder = "my_images"
test_ds = torchvision.datasets.ImageFolder(root=test_folder, transform=transformations)
test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True)

# %% create model instance
model = models.inception_v3(pretrained=True)
model.eval()  # Set the model to evaluation mode

#%%
for _, (X_batch, y_batch) in enumerate(test_loader):
    # forward pass
    with torch.no_grad():
        # Ensure the input size is correct for Inception v3
        if X_batch.size(2) != 299 or X_batch.size(3) != 299:
            X_batch = nn.functional.interpolate(X_batch, size=(299, 299), mode='bilinear', align_corners=False)
        y_test_pred_batch = model(X_batch)
        
# %%
y_test_pred_batch
# %% imagenet labels
file_path = "imagenet1000_clsidx_to_labels.txt"

# load as txt
with open(file_path, 'r') as f:
    imagenet_labels = f.readlines()

# %%
np.argmax(y_test_pred_batch.numpy())

# %%
imagenet_labels[np.argmax(y_test_pred_batch.numpy())]

# %%

