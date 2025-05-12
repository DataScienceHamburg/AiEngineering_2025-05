# %% packages
from ucimlrepo import fetch_ucirepo
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
 
# %% Constants
EPOCHS =100
LR = 0.01
BATCH_SIZE = 32
 
# %% fetch dataset
mushroom = fetch_ucirepo(id=73)
 
# %% prepare data
# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets
 
# metadata
print(mushroom.metadata)
 
# variable information
print(mushroom.variables)
 
# %% one-hot encoding
X_encoded = pd.get_dummies(X, dtype=int)
X_encoded
y_encoded = pd.get_dummies(y, drop_first=True, dtype=float)
#y_encoded = [1 if i == 'p' else 0 for i in np.array(y)]
#y_encoded = np.array(y_encoded)
 
# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
 
# %% data as Torch Tensors
X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
X_test_torch = torch.tensor(X_test.values, dtype=torch.float32)
y_train_torch = torch.from_numpy(y_train.values).reshape(-1,1)
y_test_torch = torch.from_numpy(y_test.values).reshape(-1,1)
 
# %% Dataset and DataLoader
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
 
train_ds = MyDataset(X=X_train_torch, y=y_train_torch)
test_ds = MyDataset(X=X_test_torch, y=y_test_torch)
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)
 
# %% model class
class MushroomModel(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super().__init__()
        self.input = nn.Linear(input_nodes, hidden_nodes)
        self.hidden = nn.Linear(hidden_nodes, hidden_nodes)
        self.output = nn.Linear(hidden_nodes, output_nodes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
 
# %% model instance
input_nodes = X_encoded.shape[1]
hidden_nodes = 116
output_nodes = y_encoded.shape[1] #len(np.unique(y))
model = MushroomModel(input_nodes, hidden_nodes, output_nodes)
model
 
# %% loss function
loss_function = nn.BCELoss()
 
# %% optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
 
# %% training loop
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for _, (X_batch, y_batch) in enumerate(train_loader):
        # initialize gradient
        optimizer.zero_grad()
 
        # forward pass (prediction)
        y_pred_batch = model(X_batch)
 
        # loss calculation
        loss = loss_function(y_pred_batch, y_batch.float())
        loss_epoch += loss.item()
 
        # backward pass
        loss.backward()
 
        # update weights
        optimizer.step()
 
    # add loss
    losses.append(loss_epoch)
    print(f"Epoch: {epoch}, Loss: {loss_epoch}")
 
# %% plot loss
sns.lineplot(x=range(len(losses)), y=losses)
 
# %% model validation
y_pred_test = []
y_true_test = []
with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(test_loader):
        y_pred_batch = model(X_batch)
        y_pred_test.extend(y_pred_batch.data.numpy())
        y_true_test.extend(y_batch.data.numpy())
    y_pred_test = np.array(y_pred_test)
    y_true_test = np.array(y_true_test)
    
y_pred_test_classes = (y_pred_test > 0.5).astype(int)
y_true_test_classes = (y_true_test > 0.5).astype(int)
 
# %%
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y_true_test_classes, y_pred_test_classes), annot=True, cmap='Blues')
 