import os
import zipfile
from pathlib import Path

import mlflow
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
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
    transforms.Resize((384, 384)), 
    transforms.RandomResizedCrop(384, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_dataset = datasets.ImageFolder(dataset_root)
num_classes = len(base_dataset.classes)

if num_classes < 2:
    raise ValueError("Dataset must contain at least 2 classes")

train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size

train_idx, val_idx = random_split(
    range(len(base_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset_full = datasets.ImageFolder(dataset_root, transform=train_transform)
val_dataset_full = datasets.ImageFolder(dataset_root, transform=val_transform)

train_dataset = Subset(train_dataset_full, train_idx.indices)
val_dataset = Subset(val_dataset_full, val_idx.indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
model = models.efficientnet_b3(weights=weights)

for param in model.parameters():
    param.requires_grad = False

for param in model.features[-4:].parameters():
    param.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 512),
    nn.GELU(),
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(256, num_classes)
)

for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
learning_rate = 1e-4  
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=learning_rate,
    weight_decay=1e-4,
    eps=1e-8
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)

epochs = 15
best_val_acc = 0.0
best_epoch = 0

with mlflow.start_run() as run:
    mlflow.log_param("model", "efficientnet_b3")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("dataset_root", str(dataset_root))
    mlflow.log_param("image_size", 384)
    mlflow.log_param("unfrozen_layers", "features[-4:] + classifier")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_loss = running_loss / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        scheduler.step(epoch)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)

        print(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, "best_efficientnet_b3.pth")

    mlflow.log_metric("best_accuracy", best_val_acc)
    mlflow.log_param("best_epoch", best_epoch)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id}")
    print(f"Best Accuracy: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")
