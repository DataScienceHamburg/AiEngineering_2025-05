#%%
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

#%% fetch dataset 
mushroom = fetch_ucirepo(id=73) 
  
#%% data (as pandas dataframes) 
X = mushroom.data.features 
y = mushroom.data.targets 
  
# metadata 
print(mushroom.metadata) 
  
# variable information 
print(mushroom.variables) 

# %% One Hot Encoding
X_encoded = np.array(pd.get_dummies(X, dtype=int))
X_encoded

#%%
y_encoded = [1 if i == 'p' else 0 for i in np.array(y)]
y_encoded = np.array(y_encoded)

# %%
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
        self.y = torch.from_numpy(y).float()  # Ensure y is float for BCELoss
    
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
hidden_nodes = 8
output_nodes = 1  # Change to 1 for binary classification

model = MushroomModel(input_nodes=input_nodes, hidden_nodes=hidden_nodes, output_nodes=output_nodes)

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
        loss = loss_function(y_pred_train, y_batch.view(-1, 1))  # Ensure y_batch is reshaped for BCELoss
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

# %% model validation
y_pred_test = []
y_true_test = []
with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(test_loader):
        y_pred_test_batch = model(X_batch.float()).data.numpy()
        y_pred_test_batch_classes = (y_pred_test_batch > 0.5).astype(int)  # Use threshold for binary classification
        y_pred_test.extend(y_pred_test_batch_classes)
        y_true_test.extend(y_batch.data.numpy())
y_pred_test   

# %%
y_pred_test = [int(value) for value in y_pred_test]
y_true_test = [int(value) for value in y_true_test]
y_pred_test

# %% confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true=y_true_test, y_pred=y_pred_test)
sns.heatmap(cm, annot=True, cmap='Blues')
# %%
accuracy_score(y_true=y_true_test, y_pred=y_pred_test)
# %% test plan

df_test_plan = pd.DataFrame({'LR': [0.01, 0.001, 0.01, 0.001, ],
                             'EPOCHS': [20, 20, 40, 40],
                             })
df_test_plan['accuracy'] = 0
df_test_plan

# %%
