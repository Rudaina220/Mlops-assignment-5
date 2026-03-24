import os
import zipfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI is not set")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Assignment5_Mlops")

zip_path = Path("Agriculture.zip")
extract_root = Path("data")

if not zip_path.exists():
    raise FileNotFoundError(f"{zip_path} not found")

if not extract_root.exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_root)

def find_imagefolder_root(base_path: Path):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    candidates = []

    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        class_dirs = []
        for d in dirs:
            dpath = root_path / d
            has_image = False
            for _, _, subfiles in os.walk(dpath):
                if any(Path(f).suffix.lower() in image_exts for f in subfiles):
                    has_image = True
                    break
            if has_image:
                class_dirs.append(d)
        if len(class_dirs) >= 2:
            candidates.append(root_path)

    if not candidates:
        raise FileNotFoundError("Could not find an ImageFolder-style dataset after extracting Agriculture.zip")

    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]

dataset_root = find_imagefolder_root(extract_root)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

base_dataset = datasets.ImageFolder(dataset_root)

num_classes = len(base_dataset.classes)
if num_classes < 2:
    raise ValueError("Dataset must contain at least 2 classes")

train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size
train_dataset, val_dataset = random_split(
    base_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes)
)

for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
epochs = 15
best_val_acc = 0.0

with mlflow.start_run() as run:
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("dataset_root", str(dataset_root))
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("model", "resnet18_pretrained_frozen_backbone")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("accuracy", val_acc, step=epoch)
        mlflow.log_metric("best_accuracy", best_val_acc, step=epoch)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {best_val_acc}")

    mlflow.log_metric("final_accuracy", best_val_acc)
