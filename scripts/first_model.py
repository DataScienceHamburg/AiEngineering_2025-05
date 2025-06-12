#%% package
import torch
import kagglehub
import os 
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# Download latest version
path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset")

print("Path to dataset files:", path)
#%% data import
file_name = "enhanced_anxiety_dataset.csv"
complete_file_path = os.path.join(path, file_name)
anxiety = pd.read_csv(complete_file_path)
# %%
anxiety
# %%
anxiety.columns

#%%
anxiety.info()

#%%
anxiety['Occupation']
# %% One-Hot-Encoding von kategorischen Features
anxiety_dummies = pd.get_dummies(anxiety, drop_first=True, dtype=int)
anxiety_dummies

# %%
anxiety_dummies.columns

#%%
anxiety_dummies.shape

#%%
sns.regplot(data=anxiety_dummies, x='Sleep Hours', y='Anxiety Level (1-10)')

#%% correlation matrix
corr_vals = anxiety[['Age', 'Sleep Hours', 'Stress Level (1-10)', 'Anxiety Level (1-10)']].corr()
sns.heatmap(corr_vals)
# %% separate independent / dependent features
X = np.array(anxiety_dummies.drop(columns=['Anxiety Level (1-10)']), dtype=np.float32)
y = np.array(anxiety_dummies['Anxiety Level (1-10)'], dtype=np.float32)
print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% data scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X
# %% convert to tensors
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y).unsqueeze(1)

#%% training
w = torch.zeros(X.shape[1], 1, requires_grad=True, dtype=torch.float32)
b = torch.zeros(1, requires_grad=True, dtype=torch.float32)
print(f"w shape: {w.shape}, b shape: {b.shape}")

#%% training loop hyperparameter
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
        w -= LEARNING_RATE * w.grad # w = w - LR * w.grad
        b -= LEARNING_RATE * b.grad 

        # zero gradients
        w.grad.zero_()
        b.grad.zero_()

    # store losses
    losses_epoch.append(loss.item())

# %% visualise losses
sns.lineplot(x = list(range(EPOCHS)), y = losses_epoch)
