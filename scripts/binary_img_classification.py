#%%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.metrics import accuracy_score

#%% check if cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
device
# %% transformations
my_transformations = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# %% Hyperparameter
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

# %% dataset
train_dataset = torchvision.datasets.ImageFolder(root="data_binary/train", 
                                                 transform=my_transformations)
test_dataset = torchvision.datasets.ImageFolder(root="data_binary/test",
                                                transform=my_transformations)

# %% dataloader
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset, 
                          batch_size=BATCH_SIZE,
                          shuffle=True)
# %% model
class ImageClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16*6*6, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)  # [BS, 6, 30, 30]
        x = self.relu(x)
        x = self.pool(x)  # [BS, 6, 15, 15]
        x = self.conv2(x)  #  [BS, 16, 13, 13]
        x = self.relu(x)
        x = self.pool(x)  # [BS, 16, 6, 6]
        x = x.reshape(x.size(0), -1)  # [BS, 16*6*6]
        x = self.fc1(x) # [BS, 64]
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        # x  # output [BS, 1]
        return x
    
model = ImageClassificationModel().to(device)
# dummy_input = torch.randn(1, 1, 32, 32)  # (BS, C, H, W)
# model(dummy_input).shape
# %% optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), 
                            lr = LEARNING_RATE)

loss_fun = nn.BCELoss()
# %% training loop
train_losses = []
for epoch in range(EPOCHS):
    running_loss = 0
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
        # move data to device
        X_train_batch = X_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)
        
        # zero gradients
        optimizer.zero_grad()

        # forward pass
        y_train_pred = model(X_train_batch)

        # loss calc
        loss = loss_fun(y_train_pred, y_train_batch.reshape(-1, 1).float())

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # extract losses
        running_loss += loss.item()
    train_losses.append(running_loss)
    print(f"Epoch {epoch}: Train Loss: {running_loss}")
# %%
import seaborn as sns
sns.lineplot(x=range(EPOCHS), y=train_losses)

#%% test loop
y_test_true, y_test_pred = [], []
for i, (X_test_batch, y_test_batch) in enumerate(test_loader):
    with torch.no_grad():
        y_test_pred_batch = model(X_test_batch)
        y_test_true.extend(y_test_batch.detach().numpy().tolist())
        y_test_pred.extend(y_test_pred_batch.detach().numpy().tolist())
# %%
from sklearn.metrics import confusion_matrix
threshold = 0.5
y_test_pred_class = [1 if float(i[0]) > threshold else 0 for i in y_test_pred]

cm = confusion_matrix(y_pred=y_test_pred_class, y_true=y_test_true)
sns.heatmap(cm, annot=True, fmt="d")
# %%
accuracy = accuracy_score(y_test_pred_class, y_test_true)

# %%
accuracy
# %%
