#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np

#%% data import
iris = load_iris()

#%% separation of independent and dependent feature
X = iris['data']
y = iris['target']

#%% Hyperparameter
EPOCHS = 200
BATCH_SIZE = 8
LR = 0.001

#%% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Dataset class 
class IrisDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
        
    
train_ds = IrisDataset(X=X_train, y=y_train)
test_ds = IrisDataset(X=X_test, y=y_test)

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
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

# %% model instance
input_nodes = X.shape[1]
hidden_nodes = 8
output_nodes = len(np.unique(y))

model = IrisModel(input_nodes=input_nodes, hidden_nodes=hidden_nodes, output_nodes=output_nodes)

# %% loss function
loss_function = nn.CrossEntropyLoss()

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
        y_pred_train = model(X_batch)
        
        # calc losses
        loss = loss_function(y_pred_train, y_batch.long())
        epoch_loss += loss.item()
        
        # calc gradients
        loss.backward()
        
        # update weights
        optimizer.step()
    
    # append loss list
    losses.append(epoch_loss)
    
#%% visualise losses
sns.lineplot(x=range(len(losses)), y=losses)

# %% model validation
y_pred_test = []
y_true_test = []
with torch.no_grad():
    for i, (X_batch, y_batch) in enumerate(test_loader):
        y_pred_test_batch = model(X_batch).data.numpy()
        y_pred_test_batch_classes = np.argmax(y_pred_test_batch, axis=1)
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