#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%% Constants
EPOCHS =100
LR = 0.01
BATCH_SIZE = 32
# %%
coffee_file = '../data/coffee_shop_revenue.csv'
coffee = pd.read_csv(coffee_file)
coffee.head()


#%% separate independent from dependent features
# use everything except daily revenue as independent features
X = np.array(coffee.drop(columns=['Daily_Revenue']))
# X = np.array(coffee[['Number_of_Customers_Per_Day']])
y = np.array(coffee['Daily_Revenue']).reshape(-1, 1)
X.shape

#%% create train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Deep Learning
# convert X and y to torch tensors
X_train_torch = torch.from_numpy(X_train).type(torch.float32)
y_train_torch = torch.from_numpy(y_train).type(torch.float32)
X_test_torch = torch.from_numpy(X_test).type(torch.float32)
y_test_torch = torch.from_numpy(y_test).type(torch.float32)

#%% Dataset 
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_ds = LinearRegressionDataset(X=X_train_torch, y=y_train_torch)
test_ds = LinearRegressionDataset(X=X_test_torch, y=y_test_torch)
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False)


#%% model class
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=20):
        super().__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x
    
#%% model instance
input_dim = X_train_torch.shape[1]
output_dim = y_train_torch.shape[1]
hidden_dim = 50
model = RegressionModel(input_size=input_dim, output_size=output_dim, hidden_size=hidden_dim)

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

# %% test model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_torch)
    
# %%
# Convert tensors to numpy arrays and flatten them to 1D arrays
y_pred_np = y_pred.numpy().flatten()
y_test_np = y_test_torch.numpy().flatten()

# %%
# Create scatter plot with 1D arrays
sns.scatterplot(x=y_pred_np, y=y_test_np)
# color red for regression line
sns.regplot(x=y_pred_np, y=y_test_np, scatter=False, line_kws={'color': 'red'})

#%% extract R2
from sklearn.metrics import r2_score
r2_score(y_test_np, y_pred_np)

#%% visualisation of predictions as gaussian distribution
import matplotlib.pyplot as plt
df_res = pd.DataFrame({'y_pred': y_pred_np, 'y_test': y_test_np})
df_res['error_percentage'] = (df_res['y_pred'] - df_res['y_test']) / df_res['y_test'] * 100
sns.kdeplot(df_res['error_percentage'], fill=True)
plt.title('Density Plot of Error Percentage')
plt.xlabel('Error Percentage')
plt.ylabel('Density')
plt.show()

#%% derive standard deviation
df_res['error_percentage'].std()

#%% model 1 (6, 50, 1)
# all features 0.845
# only Daily_Customers 0.505

#%% model 2 (6, 20, 20, 1)
# all features 0.836
# only Daily_Customers 

#%% model 3 (6, 30, 30, 1)
# all features 0.841
# only Daily_Customers 0.505

#%% model 4 (6, 10, 1)
# all features: 0.810 



# %% create sklearn model predictions
from sklearn.linear_model import LinearRegression
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)

# %%
r2_score(y_test_np, y_pred_sklearn)

# %% state dictionary
model.state_dict()
# %% save model parameters
model_weights_path = "saved_model/reg_model_01.pt"
torch.save(model.state_dict(), model_weights_path)

# %%
model_restored = RegressionModel(input_size=input_dim, output_size=output_dim, hidden_size=hidden_dim)

model_restored.load_state_dict(torch.load(model_weights_path))
model_restored.state_dict()
# %%
