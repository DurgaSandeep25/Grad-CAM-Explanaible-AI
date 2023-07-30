
"""
## Gradient Class Activation Map
Understanding Grad-CAM - Explainable CNN through a real-world example
- Dataset: vegetable dataset from Github
- Objective: Multi-class classification

Note: There are good open-source packages already available which can be used in your exisiting pipelines

"""

!git clone https://github.com/parth1620/GradCAM-Dataset.git
!pip install -U git+https://github.com/albumentations-team/albumentations
!pip install --upgrade opencv-contrib-python

"""# Import Required Packages"""

import sys
sys.path.append('/content/GradCAM-Dataset')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import utils

"""# Configurations"""

# Following will be configured based on your system path and requirements

# Data Path
CSV_FILE = '/content/GradCAM-Dataset/train.csv'
DATA_DIR = '/content/GradCAM-Dataset/'

# GPU
DEVICE = 'cuda'

# Model related fields
BATCH_SIZE=16
LR = 10**-3
EPOCHS=20

"""### Data
- Mushroom : Label 2
- Egg Plant : Label 1
- Cucumber : Label 0
"""

data = pd.read_csv(CSV_FILE)
train_df, valid_df = train_test_split(data, test_size=0.2, random_state=42)
print("Train Data Size : ", train_df.shape[0])
print("Test Data Size : ", valid_df.shape[0])

"""# Augmentations"""

# mean and std are mean and std of imagenet - its a common-practice but if your custom dataset is something "special" like medical images - plz calcualte mean and std from your dataset
train_augs = A.Compose([
    A.Rotate(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# On Validation, only normalization is done
valid_augs = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

"""# Load Image Dataset"""

# this can be pushed into the utils.py file (For now keeping it here)
class ImageDataset(Dataset):

    def __init__(self, df, data_dir = None, augs = None,):
        self.df = df
        self.augs = augs
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = self.data_dir + row.img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = row.label

        if self.augs:
            data = self.augs(image = img)
            img = data['image']

        # Re-arranges RGB - (not required in general)
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, label

trainset = ImageDataset(train_df, augs=train_augs, data_dir = DATA_DIR)
validset = ImageDataset(valid_df, augs=valid_augs, data_dir = DATA_DIR)

print(f"No. of examples in the {len(trainset)}")
print(f"No. of examples in the {len(validset)}")

"""# Load Dataset into Batches"""

trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(dataset=validset, batch_size=BATCH_SIZE, shuffle=True)

print(f"No. of batches in trainloader : {len(trainloader)}")
print(f"No. of batches in validloader : {len(validloader)}")

for image, labels in trainloader:
  break

print(f"One batch image shape : {image.shape}")
print(f"One batch label shape : {labels.shape}")

"""# Create Model"""

class ImageModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.feature_extractor = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels= 16, kernel_size=(5,5), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(4,4), stride=2),

        nn.Conv2d(in_channels=16, out_channels= 16, kernel_size=(5,5), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(4,4), stride=2),

        nn.Conv2d(in_channels=16, out_channels= 32, kernel_size=(5,5), padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(4,4), stride=2),

        nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=(5,5), padding=1),
        nn.ReLU()
    )

    self.maxpool = nn.MaxPool2d(kernel_size=(4,4), stride=2)

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(6400, 2048),
        nn.ReLU(),
        nn.Linear(2048, 3)
    )

    self.gradient = None

  def activations_hook(self, grad):
    self.gradient = grad

  def forward(self, images):

    x = self.feature_extractor(images) # this is used as activation map

    h = x.register_hook(self.activations_hook)

    x = self.maxpool(x)
    x = self.classifier(x)

    return x

  def get_activation_grad(self): # this gives weights of activation
    return self.gradient

  def get_activation(self, x): # this gives activation map
    return self.feature_extractor(x)

model = ImageModel()
model.to(DEVICE)

"""# Create Train and Eval function"""

# In Torch, we use cross entropy as criterion which requires logits (before doing softmax) and true labels as input

def train_function(dataloader, model, optimizer, criterion):
  model.train()
  total_loss=0.0

  for images, labels in tqdm(dataloader):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

  return total_loss/len(dataloader)

def valid_function(dataloader, model, criterion):
  model.eval()
  total_loss=0.0

  for images, labels in tqdm(dataloader):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    logits = model(images)
    loss = criterion(logits, labels)
    total_loss += loss.item()
  return total_loss/len(dataloader)

"""# Training Loop"""

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

train_loss_lis = []
valid_loss_lis = []
best_valid_loss = float('inf')

for i in range(EPOCHS):

  train_loss = train_function(trainloader, model, optimizer, criterion)
  valid_loss = valid_function(validloader, model, criterion)

  train_loss_lis.append(train_loss)
  valid_loss_lis.append(valid_loss)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_weights.pt')
    best_valid_loss = valid_loss
    print("Saved Successfully")


  print("Epoch : {}, Train Loss : {}, Valid Loss : {}".format(i, train_loss, valid_loss))

plt.plot(np.arange(len(train_loss_lis)), train_loss_lis)
plt.plot(np.arange(len(valid_loss_lis)), valid_loss_lis)
plt.show()

"""# Inference - Explainability of our model : Get GradCAM"""
def get_gradcam(model, image, label, size):

  label.backward()

  gradients = model.get_activation_grad()
  pooled_gradients = torch.mean(gradients, dim=[0,2,3])
  activations = model.get_activation(image).detach()

  for i in range(activations.shape[1]):
    activations[:, i, :, :] *= pooled_gradients[i]
  heatmap = torch.mean(activations, dim=1).squeeze().cpu()
  heatmap = nn.ReLU()(heatmap)
  heatmap /= torch.max(heatmap)
  heatmap = cv2.resize(heatmap.numpy(), (size, size))

  return heatmap

image, label = validset[4]
denorm_image = image.permute(1,2,0)*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

image = image.unsqueeze(0).to(DEVICE)
pred = model(image)
heatmap = get_gradcam(model, image, pred[0][1], size=227)

image, label = validset[4]
denorm_image = image.permute(1,2,0)*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

image = image.unsqueeze(0).to(DEVICE)
pred = model(image)
heatmap = get_gradcam(model, image, pred[0][1], size=227)
utils.plot_heatmap(denorm_image, pred, heatmap)

image, label = validset[4]
denorm_image = image.permute(1,2,0)*np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

image = image.unsqueeze(0).to(DEVICE)
pred = model(image)
heatmap = get_gradcam(model, image, pred[0][2], size=227)

utils.plot_heatmap(denorm_image, pred, heatmap)









