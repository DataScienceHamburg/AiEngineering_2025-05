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


# convert X and y to torch tensors
X_torch = torch.from_numpy(X).type(torch.float32)
y_torch = torch.from_numpy(y).type(torch.float32)

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
EPOCHS =20000
LR = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%% training loop
losses = []
for epoch in range(EPOCHS):
    # set gradient to zero
    optimizer.zero_grad()
    
    # forward pass
    y_pred = model(X_torch)
    
    # loss calculation
    y_true = y_torch
    loss = loss_function(y_pred, y_true)
    
    # backward pass
    loss.backward()
    
    # update weights
    optimizer.step()
    
    # add losses
    losses.append(float(loss.data))
    # print(f"Epoch: {epoch}, Loss: {loss.item()}")
            

# %% loss-visualisation
sns.lineplot(x=range(len(losses)), y=losses)
#%% extract slope and bias
for name, param in model.named_parameters():
    print(param.data)