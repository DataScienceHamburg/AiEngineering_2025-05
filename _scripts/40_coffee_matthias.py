#%% packages
import graphlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
# import hiddenlayer
 
#%% data import
#cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars_file= "../data/coffee_shop_revenue.csv"
cars = pd.read_csv(cars_file)
cars.head()
 
#%% visualise the model
#sns.scatterplot(x='wt', y='mpg', data=cars)
#sns.regplot(x='wt', y='mpg', data=cars)
 
#%% convert data to tensor
X_list = cars.iloc[:, 0:6]
X_np = np.array(X_list, dtype=np.float32)
y_list = cars.iloc[:, -1]

y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)
print(X.shape)
print(y_true.shape)
#%%
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=50):
        super(LinearRegressionTorch, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_hidden = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_dim)
 
   
 
   
    def forward(self, x):
        x = self.linear_in(x)
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return x
 
input_dim = 6
output_dim = 1
hidden_size = 10
model = LinearRegressionTorch(input_size=input_dim, output_size=output_dim, hidden_size=hidden_size)
model.train()
 
# %% Mean Squared Error
loss_fun = nn.MSELoss()
 
#%% Optimizer
learning_rate = 0.002

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
#%% perform training
losses = []
slope, bias = [], []
NUM_EPOCHS = 1000
BATCH_SIZE = 32
for epoch in range(NUM_EPOCHS):
    for i in range(0, X.shape[0], BATCH_SIZE):
        # optimization
        optimizer.zero_grad()
 
        # forward pass
        y_pred = model(X[i:i+BATCH_SIZE])
 
        # compute loss
        loss = loss_fun(y_pred, y_true[i:i+BATCH_SIZE])
        losses.append(loss.item())
 
        # backprop
        loss.backward()
 
        # update weights
        optimizer.step()
   
    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])
 
 
    # store loss
    losses.append(float(loss.data))
    # print loss
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")
 
   
 
# %% visualise model training
sns.scatterplot(x=range(len(losses)), y=losses)
 
#%% visualise the bias development
sns.lineplot(x=range(NUM_EPOCHS), y=bias)
#%% visualise the slope development
sns.lineplot(x=range(NUM_EPOCHS), y=slope)
 
 
 
# %% check the result
model.eval()
y_pred = [i[0] for i in model(X).data.numpy()]
y = [i[0] for i in y_true.data.numpy()]
sns.scatterplot(x=X_list, y=y)
sns.lineplot(x=X_list, y=y_pred, color='red')
# %%
#import hiddenlayer as hl
#graph = hl.build_graph(model, X)
# %%