#%% packages
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
 
 
#%% hyperparemeter
 
BATCH_SIZE = 32
LEARNING_RATE = 0.1
EPOCHS = 400
 
 
 
 
# %% fetch dataset, target = 'num'
heart_disease = fetch_ucirepo(id=45)
 
# data (as pandas dataframes)
heart_features = heart_disease.data.features
target = heart_disease.data.targets   #.astype("category")
 
 
# metadata
print(heart_disease.metadata)
 
# variable information
print(heart_disease.variables)
 
 
# %% create categories
 
categorical_feature = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
 
for col in categorical_feature:
    heart_features[col] = heart_features[col].astype('category')
 
 
# %% drops
 
# drop rows with NAs
# heart_features_dropped = heart_features.dropna()
 
heart_features_dropped = heart_features.drop(columns = ['ca', 'thal'])
 
# %% one-hot-encoding categories
 
heart_dummies = pd.get_dummies(heart_features_dropped, drop_first=True, dtype= int)
 
heart_dummies.info()
 
 
 
# %% seperate independent/ dependent feature (y = target)
 
X = np.array(heart_dummies, dtype = np.float32)
y = np.array(target, dtype = np.float32)
 
print(f"X shape: {X.shape}, y shape: {y.shape}")
 
 
 
# %% train test split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=589)
 
 
# %% scale data
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
 
# %% Dataset
 
class HeartDisData(Dataset):
    def __init__(self,X,y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).type(torch.long)
 
    def __len__(self):
        return len(self.X)
   
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]    
   
train_data = HeartDisData(X = X_train_scaled, y = y_train)
test_data = HeartDisData(X = X_test_scaled, y = y_test)
 
# %% DataLoader
 
train_loader = DataLoader(dataset=train_data, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset=test_data, batch_size = BATCH_SIZE, shuffle = True)
 
 
# %% model class
 
class HeartDisModel(torch.nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.linear1 = torch.nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.linear2 = torch.nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim = 1)
 
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
 
 
# %% model instance
 
NUM_FEATURES = X.shape[1]
NUM_CLASSES = len(np.unique(y))
 
HIDDEN_FEATURES = 20
 
model = HeartDisModel(NUM_FEATURES= NUM_FEATURES, NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN_FEATURES)
 
 
# %% optimizer
 
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = torch.nn.MSELoss()
 
 
 
# %% train model
 
train_losses = []
 
for epoch in range(EPOCHS):
    loss_epoch = 0
    for X_batch, y_batch in train_loader:
 
        # zero gradients
        optimizer.zero_grad()
       
        # formward pass
        y_train_batch_pred = model(X_batch.float())
 
        # calc loss
        # Ensure y_batch is the right shape for CrossEntropyLoss (should be 1D with class indices)
        y_batch = y_batch.squeeze().long()  # Convert to 1D tensor with long dtype
        loss = loss_fn(y_train_batch_pred, y_batch)
 
        # calc grad
        loss.backward()
 
        # update parameters
        optimizer.step()
 
        # extract loss
        loss_epoch += loss.item()
 
    train_losses.append(loss_epoch)
    print(f"epoch = {epoch}, train loss: {loss_epoch}")
 
# %%
 
sns.lineplot(x = list(range(EPOCHS)), y = train_losses, label = 'train')
 
# %% test loop
 
test_losses = []
y_test_true = []
y_test_pred_all = []

with torch.no_grad():
   
    for X_batch, y_batch in test_loader:
        y_test_pred = model(X_batch.float())
        y_test_pred_class = torch.max(y_test_pred, 1).indices
        y_test_true.extend(y_batch.detach().numpy().flatten().tolist())
        y_test_pred_all.extend(y_test_pred_class.detach().numpy().flatten().tolist())
 
 
# %% test prediction
 
accuracy_score(y_pred=y_test_pred_all, y_true=y_test_true)
 
# %% Baseline classifier
 
from sklearn.dummy import DummyClassifier
 
dummy_cls = DummyClassifier(strategy='most_frequent')
dummy_cls.fit(X_train, y_train)
dummy_cls.score(X_test, y_test)
 
 
# %% confusion matrix
 
from sklearn.metrics import confusion_matrix
 
cm = confusion_matrix(y_pred=y_test_pred_all, y_true = y_test_true)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(NUM_CLASSES), yticklabels=range(NUM_CLASSES))
 
# %%