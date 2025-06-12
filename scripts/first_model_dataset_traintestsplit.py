#%% package
import torch
import kagglehub
import os 
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
# train test split
from sklearn.model_selection import train_test_split


#%% training loop hyperparameter
HIDDEN_SIZE = 50
EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 1024
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
y = np.array(anxiety_dummies[['Anxiety Level (1-10)']], dtype=np.float32)
print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Dataset 
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#%% Dataloader
train_dataset = LinearRegressionDataset(X=X_train_scaled, y=y_train)
test_dataset = LinearRegressionDataset(X=X_test_scaled, y=y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

#%% training
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LinearRegression, self).__init__()
        self.linear_in = torch.nn.Linear(input_size, hidden_size)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_out = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x
#%% model instance
model = LinearRegression(input_size=X.shape[1], output_size=y.shape[1], hidden_size=HIDDEN_SIZE)

#%% Optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()
#%% training loop
train_losses_epoch, test_losses_epoch = [], []
for epoch in range(EPOCHS):
    train_loss_epoch, test_loss_epoch, num_batches_train, num_batches_test = 0, 0, 0, 0
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
        num_batches_train += 1
    # store losses
    train_losses_epoch.append(train_loss_epoch / num_batches_train)
    # Model Validation
    for j, (X_test_batch, y_test_batch) in enumerate(test_loader):
        with torch.no_grad():
            y_test_pred = model(X_test_batch)
            loss = loss_fn(y_test_pred, y_test_batch)
            test_loss_epoch += loss.item()
            num_batches_test += 1
    test_losses_epoch.append(test_loss_epoch / num_batches_test)

    print(f"Epoch: {epoch}, current train loss: {train_loss_epoch / num_batches_train}, current test loss: {test_loss_epoch / num_batches_test}")


# %% visualise losses
sns.lineplot(x = list(range(EPOCHS)), y = train_losses_epoch)
sns.lineplot(x = list(range(EPOCHS)), y = test_losses_epoch, color='red')

# %% predict
with torch.no_grad():
    # use trainloader to predict
    y_train_pred, y_true = [], []
    for X_train_batch, y_train_batch in train_loader:
        y_train_pred.extend(model(X_train_batch).detach().numpy().flatten().tolist())
        y_true.extend(y_train_batch)
    
# %% calc correlation coefficient
from sklearn.metrics import r2_score
r2 = r2_score(y_pred=y_train_pred, y_true=y_true)
print(f"R-squared: {r2}")
#%%
# %% model state dictionary
model.state_dict()
# %% save state dictionary
torch.save(model.state_dict(), 'model1_dict.pt')



