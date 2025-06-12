# %% [markdown]
# ### imports
 
# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
 
# %% [markdown]
# ### GPU freimachen
 
# %%
# Nach den Import-Anweisungen
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # GPU-Speicher freigeben
 
# %% [markdown]
# ### GPU nutzen
 
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# %%
# img = Image.open("C:\\Users\\lvettel\\Nextcloud\\ArbeitsVZ\\AiEngineering_2025-05\\D03_ComputerVision\\BinaryClassification\\data\\test\\Positive\\00504.jpg")
# img
 
#Originalmaße: 592x500
 
# %% [markdown]
# ### Beispieltransformierung
 
# %%
preprocess_steps = transforms.Compose([
    #transforms.Resize((224, 224))
    transforms.Resize(256),
    transforms.RandomRotation(30, fill=128),  # Moderaterer Winkel mit Grau-Füllung
    transforms.CenterCrop(224),  # Beschneidet ggf. schwarze Ränder
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ElasticTransform(alpha=50.0, sigma=5.0),
    #transforms.ToTensor(),
])
# preprocess_steps(img)
 
# %% [markdown]
# ### tatsächliche Transformierung
 
# %%
my_transform = transforms.Compose([
    transforms.Resize((32)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
# my_transform(img)
 
# %% [markdown]
# ### Hyperparameter
 
# %%
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
 
# %% [markdown]
# ### Dataset
 
# %%
# Korrigierte Version mit Transformation
train_dataset = torchvision.datasets.ImageFolder(
    "data_binary\\train",
    transform=my_transform  # Transformation hinzugefügt
)
   
test_dataset = torchvision.datasets.ImageFolder(
    "data_binary\\test",
    transform=my_transform  # Transformation hinzugefügt
)
 
# %% [markdown]
# ### Dataset
 
# %%
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
 
# %% [markdown]
# ### Model
 
# %%
class ImageClassificationModel(nn.Module):
    def __init__(self):
        super(ImageClassificationModel, self).__init__()  # Korrekte Super-Klassen-Initialisierung
       
        self.conv1 = nn.Conv2d(
            in_channels=1,     # Anzahl der Eingabekanäle - 1 für Graustufenbilder
            out_channels=6,    # Anzahl der Feature Maps im ersten Layer
            kernel_size=3,     # 3×3 Filtergröße
            stride=1,          # Schrittweite 1 Pixel
            padding=1          # Padding erhält die räumliche Dimension
        )
       
        # Korrektur: in_channels muss mit out_channels von conv1 übereinstimmen (6 statt 166)
        self.conv2 = nn.Conv2d(
            in_channels=6,     # Muss mit out_channels von conv1 übereinstimmen
            out_channels=16,   # Mehr Feature Maps in tieferen Schichten
            kernel_size=3,     # 3×3 Filtergröße
            padding=1          # Padding erhält die räumliche Dimension
        )
       
        self.pool = nn.MaxPool2d(
            kernel_size=2,     # 2×2 Pooling-Fenster
        )
        self.fc1 = nn.Linear(16 * 8 * 8, 64) # 16 Feature Maps, 6×6 nach 2× MaxPooling
        self.fc2 = nn.Linear(64, 1) # 16 Feature Maps, 6×6 nach 2× MaxPooling
        self.relu = nn.ReLU()
       
        # Berechnung der Eingabegröße für den vollverknüpften Layer
        # Nach 2× MaxPooling wird die Bildgröße auf 32/2/2 = 8×8 reduziert
        # Mit 16 Feature Maps haben wir 16*8*8 = 1024 Features
        # self.fc = nn.Linear(16 * 8 * 8, 2)  # 2 Ausgabeklassen für binäre Klassifikation
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
       
        # Korrektur: Tensor flatten mit view oder flatten
        x = x.view(x.size(0), -1)  # Flatten, behält Batch-Dimension
       
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
 
model = ImageClassificationModel()
model = model.to(device)  # Diese Zeile fehlt in Ihrem Code
#print(model(dummy_input).shape)  # Ausgabe der Modellvorhersage
 
# %% [markdown]
# ### optimizer & loss
 
# %%
#optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# später Adam ausprobieren
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss für binäre Klassifikation
 
 
 
# %% [markdown]
# ### training
 
# %%
# Vor Beginn der Trainingsschleife (vor dem "for epoch in range(EPOCHS):")
train_losses = []  # Liste zur Speicherung aller Verluste über alle Epochen
 
for epoch in range(EPOCHS):
    model.train()  # Setzt das Modell in den Trainingsmodus
    running_loss = 0
   
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
        # Erst Daten auf GPU verschieben
        X_train_batch = X_train_batch.to(device)
        y_train_batch = y_train_batch.float().view(-1, 1).to(device)
       
        # Dann Berechnungen durchführen
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch)
        loss = loss_function(y_train_pred, y_train_batch)
        loss.backward()
        optimizer.step()
       
        running_loss += loss.item()
   
    # Speichern der Verluste für diese Epoche
    train_losses.append(running_loss)
   
    # Ausgabe des durchschnittlichen Verlusts
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(running_loss)}")
 
# %%
sns.lineplot(x=range(len(train_losses)), y=train_losses)
 
 
# %%
y_test_true, y_test_pred = [], []
for i, (X_test_batch, y_test_batch) in enumerate(test_loader):
    with torch.no_grad():
        # Daten auf GPU verschieben
        X_test_batch = X_test_batch.to(device)
        y_test_batch = y_test_batch.float().view(-1, 1).to(device)
       
        y_test_pred_batch = model(X_test_batch)
        y_test_true.extend(y_test_batch.detach().to('cpu').numpy().tolist())
        y_test_pred.extend(y_test_pred_batch.detach().to('cpu').numpy().tolist())
 
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test_true, np.round(y_test_pred))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
 