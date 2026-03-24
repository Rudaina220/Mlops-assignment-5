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
        raise FileNotFoundError("Could not find an ImageFolder-style dataset")

    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]

dataset_root = find_imagefolder_root(extract_root)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3), 
    transforms.RandomRotation(20),  
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_dataset = datasets.ImageFolder(dataset_root)
num_classes = len(base_dataset.classes)

train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size

train_idx, val_idx = random_split(
    range(len(base_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = Subset(datasets.ImageFolder(dataset_root, transform=train_transform), train_idx.indices)
val_dataset = Subset(datasets.ImageFolder(dataset_root, transform=val_transform), val_idx.indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4,  
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, epochs=20, steps_per_epoch=len(train_loader),
    pct_start=0.1, div_factor=25, final_div_factor=1e4
)

epochs = 5
best_val_acc = 0.0

with mlflow.start_run() as run:
    mlflow.log_param("model", "resnet50_agriculture")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 3e-4)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("image_size", 224)
    mlflow.log_param("unfrozen", "layer3+layer4+fc")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total

        mlflow.log_metrics({
            "train_loss": train_loss / train_total,
            "train_accuracy": train_acc,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_acc,
            "lr": scheduler.get_last_lr()[0]
        }, step=epoch)

        print(f"Epoch {epoch+1}/{epochs}: Train: {train_acc:.4f}, Val: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet50_agriculture.pth")

    mlflow.log_metric("best_accuracy", best_val_acc)
    
    print(f"Run ID: {run.info.run_id}")
    print(f"Best Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
