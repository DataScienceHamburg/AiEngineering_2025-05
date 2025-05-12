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
    torchvision.transforms.Resize((32, 32)),
    # torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
])
# %% Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001

# %%
train_folder = "data_concrete/train"
test_folder = "data_concrete/test"
train_ds = torchvision.datasets.ImageFolder(root=train_folder, transform=transformations)
test_ds = torchvision.datasets.ImageFolder(root=test_folder, transform=transformations)
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True)

#%% function for visualise images
def imshow(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))  # we have C, H, W-> wee need H, W, C
    plt.show()
# %% visualise sample images
data_iter = iter(train_loader)
images, labels = next(data_iter)
imshow(torchvision.utils.make_grid(images, nrow=4))
#%%
labels

#%% model
model = models.densenet121(pretrained=True)
# %%
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

#%% freeze model parameters
for params in model.parameters():
    params.requires_grad = False
# %%
model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 1)),
    ('output', nn.Sigmoid())
]))
# %% optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = nn.BCELoss()

#%% training loop
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for _, (X_batch, y_batch) in enumerate(train_loader):
        # gradients to zero
        optimizer.zero_grad()
        
        # forward pass
        y_batch_pred = model(X_batch)
        
        # calc loss
        loss = loss_fun(y_batch_pred, y_batch.reshape(-1, 1).float())
        
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        
        # losses update
        loss_epoch += loss.item()
    losses.append(loss_epoch)
    print(f"Epoch: {epoch}, Loss: {loss_epoch}")
    

# %%
import seaborn as sns
sns.lineplot(x = range(len(losses)), y= losses)

# %% Evaluate model
y_test_true = []
y_test_pred = []
for _, (X_batch, y_batch) in enumerate(test_loader):
    # forward pass
    with torch.no_grad():
        y_test_pred_batch = model(X_batch).round().numpy()
        y_test_pred.extend(y_test_pred_batch.tolist())
        y_test_true.extend(y_batch.numpy().tolist())
        

# %% confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_true, y_test_pred)
cm_normalized = confusion_matrix(y_test_true, y_test_pred, normalize='true')*100
cm_normalized = cm_normalized - 2*np.triu(cm_normalized, 1) - 2*np.tril(cm_normalized, -1)
labels = np.unique(y_test_true)
sns.heatmap(cm_normalized, xticklabels=labels, yticklabels=labels, annot=cm, fmt='.0f', vmin=-100, vmax=100, cmap='PiYG', cbar_kws={'format':'%d%%'})
#%%
accuracy_score(y_test_true, y_test_pred)

#%%
