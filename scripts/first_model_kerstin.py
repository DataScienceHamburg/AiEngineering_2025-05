 
#%% packages
import torch
import os
import pandas as pd
import kagglehub
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
 
 
# %% load data, Anxiety Level (1-10) is the target
 
 
# Download latest version
path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset")
 
print("Path to dataset files:", path)
 
# %% import data to python
 
file_name = "enhanced_anxiety_dataset.csv"
 
complete_file_path = os.path.join(path, file_name)
 
anxiety = pd.read_csv(complete_file_path)
 
# %% look at data
anxiety.info()
 
 
# %% one-hot-encoding categories
 
anxiety_dummies = pd.get_dummies(anxiety, drop_first=True, dtype= int)
 
anxiety_dummies.info()
 
 
# %% some data vis
 
sns.regplot(data = anxiety_dummies, x = 'Stress Level (1-10)', y = 'Anxiety Level (1-10)')
 
 
# %%
 
corr_vals = anxiety_dummies.corr()
 
sns.heatmap(corr_vals)
 
 
 
# %% seperate independent/ dependent feature (y = target)
 
X = np.array(anxiety_dummies.drop(columns = ['Anxiety Level (1-10)']), dtype = np.float32)
y = np.array(anxiety_dummies['Anxiety Level (1-10)'], dtype = np.float32)
 
print(f"X shape: {X.shape}, y shape: {y.shape}")
 
 
# %% scale data
 
scaler = StandardScaler()
X = scaler.fit_transform(X)
X
# %% convert to tensor
 
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
 
 
# %% training
 
w = torch.zeros(X.shape[1],1, requires_grad = True, dtype = torch.float32)
b = torch.zeros(1, requires_grad = True, dtype = torch.float32)
 
print(f"w shape: {w.shape}, b shape: {b.shape}")
 
# %% training loop hyperparameter
 
EPOCHS = 100
LEARNING_RATE = 0.01
 
losses_epoch = []
 
for epoch in range(EPOCHS):
    print(f"Epoch: {epoch}")
    # forward pass
    y_pred = torch.matmul(X_tensor, w) + b
 
    # loss calculation
    loss = torch.nn.functional.mse_loss(y_pred, y_tensor)
 
    # backward pass
    loss.backward()
 
    # weight update
    with torch.no_grad():
        w -=LEARNING_RATE * w.grad
        b -=LEARNING_RATE * b.grad
 
        # zero gradients
        w.grad.zero_()
        b.grad.zero_()
   
    # store losses
    losses_epoch.append(loss.item())
 
 
 
 
 
 
# %%
 
sns.lineplot(x = list(range(EPOCHS)), y = losses_epoch)
# %%
 
 