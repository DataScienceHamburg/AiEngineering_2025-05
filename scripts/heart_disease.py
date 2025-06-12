#%% packages
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

#%% Hyperparameters
LEARNING_RATE = 0.01
NUM_EPOCHS = 200
BATCH_SIZE = 128
HIDDEN_SIZE = 50

#%% fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
#%% data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# %%
X
# %%
y
# %% convert y to long tensor format (not one-hot encoding)
# CrossEntropyLoss expects class indices, not one-hot encoding
y = np.array(y).flatten() 

# %% encode X columns (sex, cp, fbs, restecg, exang, slope, ca, thal) as one-hot encoding
X = pd.get_dummies(X, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True, dtype=int)
# %%
X
# %%
y
# %% train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# %% Data Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %% Dataset class
class HeartDiseaseDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) 

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
train_dataset = HeartDiseaseDataset(X_train, y_train)
test_dataset = HeartDiseaseDataset(X_test, y_test)
# %% DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# %% Model class
# target feature is class 0, 1, 2, 3, 4
class HeartDiseaseModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HeartDiseaseModel, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
model = HeartDiseaseModel(input_size=X_train.shape[1], hidden_size=HIDDEN_SIZE, output_size=5)
    
# %% Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# %% Training loop
train_losses = []
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        y_pred_train = model(X_train_batch)
        loss = criterion(y_pred_train, y_train_batch)
        # backward pass
        loss.backward()
        # update weights
        optimizer.step()
        # update running loss
        running_loss += loss.item()
    # calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    # print progress
    if epoch % 50 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
# %% Plot training loss
sns.lineplot(x=range(NUM_EPOCHS), y=train_losses, label='Training Loss')
# %% Test model
model.eval()
with torch.no_grad():
    y_test_pred_all = []
    y_test_true_all = []
    for X_test_batch, y_test_batch in test_loader:
        y_pred_test = model(X_test_batch)
        y_pred_test = torch.argmax(y_pred_test, dim=1)
        y_test_pred_all.extend(y_pred_test.cpu().numpy().tolist())
        y_test_true_all.extend(y_test_batch.cpu().numpy().tolist())
# %% confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_true_all, y_test_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# %% accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_true_all, y_test_pred_all)
print(f"Accuracy: {accuracy:.4f}")
# %% dummy classifier
from sklearn.dummy import DummyClassifier
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(X_train, y_train)
y_pred_dummy = dummy_classifier.predict(X_test)
# %%
accuracy_dummy = accuracy_score(y_test_true_all, y_pred_dummy)
print(f"Dummy Accuracy: {accuracy_dummy:.4f}")

# %%
