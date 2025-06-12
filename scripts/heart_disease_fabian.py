#%% packages
from ucimlrepo import fetch_ucirepo
import torch
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
 
#%% get Data
# fetch dataset
heart_disease = fetch_ucirepo(id=45)
 
# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets
 
# metadata
print(heart_disease.metadata)
 
# variable information
print(heart_disease.variables)
 
# %% Parameters
EPOCHS = 100
LR = 0.01
HIDDEN_SIZE = 50
BATCH_SIZE = 1024
 
# %% Drop variables
reduced = X.drop(['ca', 'thal'],  axis = 1)
X = reduced
 
# %%
corr_vals = X[['age', 'sex', 'cp', 'chol' ]].corr()
sns.heatmap(corr_vals)
 
#%%split data into train an test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# %% data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
# %% Dataset
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32)
 
    def __len__(self):
        return len(self.X)
   
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
   
# %% DataLoader
train_dataset = LinearRegressionDataset(X=X_train_scaled, y=y_train)
test_dataset = LinearRegressionDataset(X=X_test_scaled, y=y_test)
 
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
 
 
# %% Training
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LinearRegression, self).__init__()
        self.linear_in = torch.nn.Linear(input_size, hidden_size)
        self.hidden_1 = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
 
 
    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.hidden_1(x)        
        x = self.relu(x)
        x = self.hidden_2(x)
        x = self. relu(x)
        x = self.linear_out(x)
        return x
 
#%% Model
model = LinearRegression(input_size = X.shape[1], output_size = y.shape[1], hidden_size= HIDDEN_SIZE)
 
#%% optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
loss_fn = torch.nn.MSELoss()
 
# %% Training Loop
train_losses_epoch, test_losses_epoch = [], []
for epoch in range(EPOCHS):
    train_loss_epoch, test_loss_epoch = 0, 0
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
   
        # forward pass
        y_train_pred = model(X_train_batch)
 
        # loss calculation
        loss = loss_fn(y_train_pred, y_train_batch)
 
        # backward pass
        loss.backward()
 
        # weight update
        optimizer.step()
 
         # zero gradients
        optimizer.zero_grad()
 
        # update loss epoch
        train_loss_epoch += loss.item()
 
 
 
    for j, (X_test_batch, y_test_batch) in enumerate (test_loader):
        with torch.no_grad():
            y_test_pred = model(X_test_batch)
            loss = torch.nn.functional.mse_loss(y_test_pred, y_test_batch)
            test_loss_epoch += loss.item()
    test_losses_epoch.append(test_loss_epoch)
 
    # store losses
    print(f"Epoch: {epoch}, current loss: {train_loss_epoch}")
    train_losses_epoch.append(train_loss_epoch)
 
# %% visualize losses
sns.lineplot(x = list(range(EPOCHS)), y = train_losses_epoch)
sns.lineplot(x = list(range(EPOCHS)), y = test_losses_epoch)