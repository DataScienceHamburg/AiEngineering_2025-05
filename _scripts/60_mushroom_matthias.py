#%%
from ucimlrepo import fetch_ucirepo
import pandas as pd
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import pandas as pd
 
mushroom=fetch_ucirepo(id=73)
 
# data (as pandas dataframes)
X=mushroom.data.features
Y=mushroom.data.targets
print(np.shape(Y))
 
# mettadata
print(mushroom.metadata)
print(mushroom.variables)
 
# one hot encoding
X_encoded=np.array(pd.get_dummies(X, dtype=int))
y_encoded = np.array([1 if i == 'p' else 0 for i in np.array(Y)], dtype=int)
#%% Hyperparameter
EPOCHS = 40
BATCH_SIZE = 8
LR = 0.001
 
#%% train test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
 
#%% Dataset class
class MushroomDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
   
    def __len__(self):
        return self.X.shape[0]
   
    def __getitem__(self, index):
        return self.X[index], self.y[index]
       
   
train_ds = MushroomDataset(X=X_train, y=y_train)
test_ds = MushroomDataset(X=X_test, y=y_test)
 
#%% DataLoader define
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE)
 
#%% model class
class IrisModel(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super().__init__()
        self.input = nn.Linear(input_nodes, hidden_nodes)
        self.hidden = nn.Linear(hidden_nodes, hidden_nodes)
        self.output = nn.Linear(hidden_nodes, output_nodes)        
        self.softmax = nn.Softmax(dim=1)
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
hidden_nodes = 8
output_nodes = 1
 
model = IrisModel(input_nodes=input_nodes, hidden_nodes=hidden_nodes, output_nodes=output_nodes)
 
# %% loss function
loss_function = nn.BCELoss()
 
# %% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr= LR)
 
# %% train loop
losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0
    for idx, (X_batch, y_batch) in enumerate(train_loader):
        # zero gradients
        optimizer.zero_grad()
       
        # forward pass
        y_pred_train = model(X_batch.float())
       
        # calc losses
        loss = loss_function(y_pred_train, y_batch.reshape(-1, 1).float())
        epoch_loss += loss.item()
       
        # calc gradients
        loss.backward()
       
        # update weights
        optimizer.step()
   
    # append loss list
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
   
#%% visualise losses
sns.lineplot(x=range(len(losses)), y=losses)
 
# %%