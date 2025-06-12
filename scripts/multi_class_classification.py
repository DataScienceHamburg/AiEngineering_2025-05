#%% packages
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np

#%% Hyperparameter
BATCH_SIZE = 32
LR = 0.1
EPOCHS = 100

# %% data import 
iris = load_iris()

#%% independent and dependent features
X = iris['data']
y = iris['target']

# %% train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# %% Dataset
# TODO: test to remove torch.from_numpy
# TODO: check if torch.long is necessary
class IrisData(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).type(torch.long)
        
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_data = IrisData(X=X_train, y= y_train)
test_data = IrisData(X=X_test, y= y_test)
# %% Dataloader
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

#%% Model Class
class IrisModel(torch.nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.linear1 = torch.nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        self.linear2 = torch.nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


# %% model instance
NUM_FEATURES = X.shape[1]
NUM_CLASSES = len(np.unique(y))
HIDDEN = 6
model = IrisModel(NUM_FEATURES=NUM_FEATURES, HIDDEN_FEATURES=HIDDEN, NUM_CLASSES=NUM_CLASSES)
# %% optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fun = torch.nn.CrossEntropyLoss()

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
        loss = loss_fun(y_train_batch_pred, y_batch)

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
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_test_pred = model(X_batch.float())
        y_test_pred_class = torch.max(y_test_pred, 1).indices
        y_test_true.extend(y_batch)
        
# %% check the performance
accuracy_score(y_pred=y_test_pred_class, y_true=y_test_true)
# %% Baseline Classifier / Naive Classifier / Dummy Classifier
# naive classifier
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train)
dummy_clf.score(X_test, y_test)
# %%
X
# %%
y
# %% confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred=y_test_pred_class, y_true=y_test_true)
# %%
