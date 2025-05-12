#%% packages
import torch
import torch.nn as nn
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

# initialize slope w and bias b
w = torch.rand(1, requires_grad=True, dtype=torch.float64)
b = torch.rand(1, requires_grad=True, dtype=torch.float64)

# convert X and y to torch tensors
X_torch = torch.from_numpy(X).type(torch.float64)
y_torch = torch.from_numpy(y).type(torch.float64)

# training loop
EPOCHS =200
LR = 0.01
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i in range(X_torch.shape[0]):
        
        # forwards pass -> predictions
        y_pred = X_torch[i] * w + b
        
        # calculate losses
        y_true = y_torch[i]
        loss_tensor = torch.pow(y_pred - y_true, 2)
        loss_epoch += loss_tensor
        # backward pass
        loss_tensor.backward()
        
        # update weights
        with torch.no_grad():
            w -= w.grad * LR
            b -= b.grad * LR
            # reset gradients
            w.grad.zero_()
            b.grad.zero_()
    
    # add to losses
    losses.append(loss_epoch.item())
    print(f"Epoch {epoch} loss: {loss_epoch.item()}")
            
    

# %% check results
print(f"Slope: {w.item()}, Bias: {b.item()}")

# %% loss-visualisation
sns.lineplot(x=range(len(losses)), y=losses)
#%%