import os
import zipfile
from pathlib import Path

import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Assignment5_Nature")

zip_path = Path("nature.zip")
extract_root = Path("data")

if not extract_root.exists():
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root)

dataset = datasets.ImageFolder(
    "data/natural_images",
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset,[train_size,val_size])

train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=32)

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features,len(dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0003)

epochs = 3

with mlflow.start_run() as run:

    for epoch in range(epochs):
        model.train()

        for imgs,labels in train_loader:
            imgs,labels = imgs.to(device),labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()

        print("Epoch",epoch,"done")

    print("Run ID:",run.info.run_id)

    with open("model_info.txt","w") as f:
        f.write(run.info.run_id)
