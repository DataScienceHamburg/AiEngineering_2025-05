#%% packages

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

import torchvision

import torchvision.transforms as transforms

import os

import numpy as np

from sklearn.metrics import accuracy_score
 
#%% hyperparameter

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 0.001
 
 
#%% transformations

my_transformations = transforms.Compose([

    transforms.Resize(32),

    transforms.Grayscale(),

    transforms.ToTensor()

])
 
#%% image loader / dataset

train_dataset = torchvision.datasets.ImageFolder(root="data_binary/train", transform=my_transformations)

test_dataset = torchvision.datasets.ImageFolder(root="data_binary/test", transform=my_transformations)
 
#%% dataloader

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
 
#%% model

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

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.relu(x)

        x = self.sigmoid(x)

        return x
 
 
#%% model

model = ImageClassificationModel()

# testcode, um dimensionen schritt für schritt oben im modell zu ermitteln

dummy_input = torch.randn(1, 1, 32, 32)

model(dummy_input).shape
 
#%% optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

loss_fn = nn.BCELoss()
 
#%% training loop

train_losses = []  

for epoch in range(EPOCHS):

    running_loss = 0

    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):

        # zero gradients

        optimizer.zero_grad()
 
        # forward pass

        y_train_pred = model(X_train_batch)
 
        # loss calc

        loss = loss_fn(y_train_pred, y_train_batch.reshape(-1, 1).float())
 
        # backward pass

        loss.backward()
 
        # update weights

        optimizer.step()
 
        # extract losses

        running_loss += loss.item()

    train_losses.append(running_loss)

    print(f"{i} Epoch {epoch}: Train loss: {running_loss}")
 
 
 
#%%
 