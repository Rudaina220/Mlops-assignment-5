import os
from pathlib import Path
from PIL import Image

import mlflow
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Assignment5_SingleFolderImages")

class SingleFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = [
            p for p in self.root_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
        ]

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 0  # single class only

        if self.transform:
            image = self.transform(image)

        return image, label

data_root = Path("images")
if not data_root.exists():
    raise FileNotFoundError("images folder not found. Make sure dvc pull ran successfully.")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

full_dataset = SingleFolderDataset(data_root, transform=train_transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.features[-2:].parameters():
    param.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, 1)
)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

epochs = 5
best_val_acc = 0.0
best_model_path = "best_model.pth"

with mlflow.start_run() as run:
    mlflow.log_param("model", "efficientnet_b0")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 3e-4)
    mlflow.log_param("num_classes", 1)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_loss += loss.item() * images.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_loss += loss.item() * images.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        mlflow.log_metrics({
            "train_loss": train_loss / train_total,
            "train_accuracy": train_acc,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_acc,
        }, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    mlflow.log_metric("best_accuracy", best_val_acc)
    mlflow.log_artifact(best_model_path)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id}")
    print(f"Best Accuracy: {best_val_acc:.4f}")
