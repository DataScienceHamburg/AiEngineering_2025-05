#%% packages
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_multilabel_classification
import seaborn as sns
import numpy as np

# %%

#%% Hyperparameter
BATCH_SIZE = 32
LR = 0.1
EPOCHS = 100

# %% data import 
multilabel_data = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)

#%% visualize data patterns with pairplot
import pandas as pd
# Create a DataFrame from the features
df = pd.DataFrame(multilabel_data[0])
df = df.iloc[: , :4]
df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

# add regression line to pairplot
sns.pairplot(df, kind='reg')
#%% independent and dependent features
X, y = multilabel_data[0], multilabel_data[1]

# %% train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# %% Dataset
class MultilabelData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).type(torch.long)
        
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_data = MultilabelData(X=X_train, y= y_train)
test_data = MultilabelData(X=X_test, y= y_test)
# %% Dataloader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

#%% Model Class
class MultilabelModel(torch.nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.linear1 = torch.nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.linear2 = torch.nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


# %% model instance
NUM_FEATURES = X.shape[1]
NUM_CLASSES = y.shape[1]
HIDDEN = 20
model = MultilabelModel(NUM_FEATURES=NUM_FEATURES, HIDDEN_FEATURES=HIDDEN, NUM_CLASSES=NUM_CLASSES)
# %% optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = torch.nn.BCEWithLogitsLoss()

# %% training loop
train_losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for X_batch, y_batch in train_loader:
        # zero gradients
        optimizer.zero_grad()
        # forward pass
        y_train_batch_pred = model(X_batch.float())

        # calc loss
        loss = loss_fun(y_train_batch_pred, y_batch.float())

        # calc gradients
        loss.backward()


        # update parameters
        optimizer.step()

        # extract loss
        loss_epoch += loss.item()
    train_losses.append(loss_epoch)
    print(f"Train Loss: {loss_epoch}")

    
# %%
sns.lineplot(x=range(EPOCHS), y=train_losses)

# %% test loop
test_losses = []
y_test_true = []
y_test_pred_all = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_test_pred = model(X_batch.float())
        print(y_test_pred)
        # For multilabel, we use threshold of 0.5 to determine positive predictions
        y_test_pred_binary = (y_test_pred > 0.5).float()
        # Store predictions and true labels
        y_test_pred_all.append(y_test_pred_binary)
        y_test_true.append(y_batch)
        
        # Calculate loss
        loss = loss_fun(y_test_pred, y_batch.float())
        test_losses.append(loss.item())

# Concatenate all batches
y_test_true = torch.cat(y_test_true, dim=0)
y_test_pred_all = torch.cat(y_test_pred_all, dim=0)
        
# %% check the performance
accuracy_score(y_pred=y_test_pred_all, y_true=y_test_true)

#%% dummy classifier from scratch
# convert [1, 1, 0] -> "[1, 1, 0]"
y_test_pred_str = [str(list(i.int().cpu().numpy())) for i in y_test_pred_all]
y_test_true_str = [str(list(i.int().cpu().numpy())) for i in y_test_true]

#%% sum if pred == true
#TODO: implement accuracy for multilabel
np.sum([1 if i == j else 0 for i, j in zip(y_test_pred_str, y_test_true_str)]) / len(y_test_pred_str)

#%%
from collections import Counter
most_common_class_combination = Counter(y_test_true_str).most_common()[0][1]
# %%
most_common_class_combination / len(y_test_true_str)
# %%
