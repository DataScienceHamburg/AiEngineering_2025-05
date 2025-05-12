#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#%% Constants
EPOCHS =2000
LR = 0.01
BATCH_SIZE = 4
# %%
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

# %% visualise wt and mpg
sns.scatterplot(data=cars, x='wt', y='mpg')
sns.regplot(data=cars, x='wt', y='mpg')

#%% separate independent from dependent features
X = np.array(cars['wt']).reshape(-1, 1)
y = np.array(cars['mpg']).reshape(-1, 1)
X.shape
#%% classical ML-model for reference
model = LinearRegression()
model.fit(X, y)
# %%
print(f"Slope: {model.coef_[0]}")
print(f"Bias: {model.intercept_}")

#%% Deep Learning


# convert X and y to torch tensors
X_torch = torch.from_numpy(X).type(torch.float32)
y_torch = torch.from_numpy(y).type(torch.float32)

#%% Dataset 
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_ds = LinearRegressionDataset(X=X_torch, y=y_torch)
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)


#%% model class
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.linear(x)
        return x
    
#%% model instance
input_dim = X_torch.shape[1]
output_dim = y_torch.shape[1]
model = RegressionModel(input_size=input_dim, output_size=output_dim)

#%% Loss function
loss_function = nn.MSELoss()

#%% Optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%% training loop
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for _, (X_batch, y_batch) in enumerate(train_loader):
        # set gradient to zero
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(X_batch)
        
        # loss calculation
        y_true = y_batch
        loss = loss_function(y_pred, y_true)
        loss_epoch += loss
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
        
    # add losses
    losses.append(float(loss_epoch.data))
    print(f"Epoch: {epoch}, Loss: {loss.item()}")
            

# %% loss-visualisation
sns.lineplot(x=range(len(losses)), y=losses)
#%% extract slope and bias
for name, param in model.named_parameters():
    print(param.data)
# %% exkurs: umgang mit enumerate trainloader
for _, dummy in enumerate(train_loader):
    print(dummy)

# %%
# %%
