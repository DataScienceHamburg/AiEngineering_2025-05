#%% packages
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# %% image transformations
transformations = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
])
# %% Hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LR = 0.001

# %%
train_folder = "data_concrete/train"
test_folder = "data_concrete/test"
train_ds = torchvision.datasets.ImageFolder(root=train_folder, transform=transformations)
test_ds = torchvision.datasets.ImageFolder(root=test_folder, transform=transformations)
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=True)

#%% function for visualise images
def imshow(img):
    img = img / 2 + 0.5
    img_np = img.numpy()
    plt.imshow(np.transpose(img_np, (1, 2, 0)))  # we have C, H, W-> wee need H, W, C
    plt.show()
# %% visualise sample images
data_iter = iter(train_loader)
images, labels = next(data_iter)
imshow(torchvision.utils.make_grid(images, nrow=2))
#%%
labels

#%% model definition
class ImageClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6272, 32)  # fully-connect (bzw. dense layer)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x
        
class ImageClassificationModel_Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32* 32, 64)  # fully-connect (bzw. dense layer)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)  
        x = self.relu(x)
        x = self.fc2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


input_img = torch.rand((1, 1, 32, 32))  # Dim: BS, C, H, W
model = ImageClassificationModel_Dense()
model(input_img).shape


# %% Exkurs: Number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

#%% model layers and sizes
# (BS, 1, 32, 32)  # image
# (BS, 6, 30, 30)  # after conv1
# (BS, 6, 15, 15)  # max pooling
# (BS, 16, 13, 13)           # after conv2
# (BS, 16, 6, 6)  # after pooling
# (BS, 16 * 6 * 6) after flatten

#  (BS, 1) output layer

#%% loss function, and optimizer
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# %% train loop
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for _, (X_batch, y_batch) in enumerate(train_loader):
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        y_batch_pred = model(X_batch)
        
        # Ensure predictions are between 0 and 1
        # y_batch_pred = torch.clamp(y_batch_pred, 0, 1)
        
        # loss calculation
        loss = loss_function(y_batch_pred, y_batch.reshape(-1, 1).float())
        loss_epoch += loss.item()
        # backward pass
        loss.backward()
        
        # update weights
        optimizer.step()
    losses.append(loss_epoch)
    print(f"Epoch: {epoch}, Loss: {loss_epoch}")

# %%
import seaborn as sns
sns.lineplot(x=range(EPOCHS), y=losses)
# %% Evaluate model
y_test_true = []
y_test_pred = []
for _, (X_batch, y_batch) in enumerate(test_loader):
    # forward pass
    with torch.no_grad():
        y_test_pred_batch = model(X_batch).round().numpy()
        y_test_pred.extend(y_test_pred_batch.tolist())
        y_test_true.extend(y_batch.numpy().tolist())
        # extract wrong predictions
        wrong_predictions = (y_test_pred_batch != y_batch.numpy())
        wrong_predictions_images = X_batch[wrong_predictions]
        wrong_predictions_labels = y_batch.numpy()[wrong_predictions]
        # display wrong predictions
        for i in range(len(wrong_predictions_images)):
            imshow(wrong_predictions_images[i])

# %% confusion matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_true, y_test_pred)
cm_normalized = confusion_matrix(y_test_true, y_test_pred, normalize='true')*100
cm_normalized = cm_normalized - 2*np.triu(cm_normalized, 1) - 2*np.tril(cm_normalized, -1)
labels = np.unique(y_test_true)
sns.heatmap(cm_normalized, xticklabels=labels, yticklabels=labels, annot=cm, fmt='.0f', vmin=-100, vmax=100, cmap='PiYG', cbar_kws={'format':'%d%%'})
#%%
accuracy_score(y_test_true, y_test_pred)
 
 
#%% Aufgabe

# 1. CNN parameter optimieren

# 2. Alternativmodell auf Basis von Dense-Layern (ca. 38k Parameter)


