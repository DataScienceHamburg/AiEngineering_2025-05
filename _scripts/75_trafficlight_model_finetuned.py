#%% packages
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import models

# %% image transformations
transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    # torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
])
# %% Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

# %%
base_path = "../data/tesla_sun_trafficlight"
ds = torchvision.datasets.ImageFolder(root=base_path, transform=transformations)
ds_train, ds_test = torch.utils.data.random_split(ds, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(dataset=ds_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=ds_test, batch_size=BATCH_SIZE, shuffle=True)


#%% function for visualise images
def imshow(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))  # we have C, H, W-> wee need H, W, C
    plt.show()
# %% visualise sample images
data_iter = iter(test_loader)
images, labels = next(data_iter)
imshow(torchvision.utils.make_grid(images, nrow=4))
#%%
labels

#%% model definition
model = models.densenet121(pretrained=True)

input_img = torch.rand((1, 3, 32, 32))  # Dim: BS, C, H, W
# model(input_img).shape

for params in model.parameters():
    params.requires_grad = False
# %%
from collections import OrderedDict
model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 4)),
    ('output', nn.Softmax(dim=1))
]))


# %% Exkurs: Number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")


#%% loss function, and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# %% train loop
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for _, (X_batch, y_batch) in enumerate(train_loader):
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        y_batch_pred = model(X_batch)
        
        
        # loss calculation
        loss = loss_function(y_batch_pred, y_batch)
        loss_epoch += loss.item()
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
    losses.append(loss_epoch)
    print(f"Epoch: {epoch}, Loss: {loss_epoch}")

# %%
import seaborn as sns
sns.lineplot(x=range(EPOCHS), y=losses)
# %% Evaluate model
y_test_true = []
y_test_pred = []
for _, (X_batch, y_batch) in enumerate(test_loader):
    # forward pass
    with torch.no_grad():
        y_test_pred_batch = model(X_batch).round().numpy()
        y_test_pred.extend(np.argmax(y_test_pred_batch, axis=1))
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
 
 
#%% Naive Classifier
# get most frequent class
most_frequent_class = np.argmax(np.bincount(y_test_true))
# calculate accuracy
accuracy_score(y_test_true, [most_frequent_class]*len(y_test_true))

# %%
# CNN with greyscale, 10 epochs: 65.4%
# CNN with RGB, 10 epochs: 75.0%
# CNN with RGB, 50 epochs: 93.0%
# CNN with RGB, 100 epochs: 95.8%


# %%
