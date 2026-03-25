import os
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
mlflow.set_experiment("Assignment5_FilmImage")


def find_imagefolder_root(base_path: Path) -> Path:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    candidates = []

    for root, dirs, _ in os.walk(base_path):
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
        raise FileNotFoundError(
            f"Could not find an ImageFolder-style dataset inside: {base_path}"
        )

    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


data_root = Path("images")
if not data_root.exists():
    raise FileNotFoundError("images folder not found. Make sure dvc pull ran successfully.")

dataset_root = find_imagefolder_root(data_root)
print(f"Using dataset root: {dataset_root}")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

base_dataset = datasets.ImageFolder(dataset_root)
num_classes = len(base_dataset.classes)

if num_classes < 2:
    raise ValueError(f"Need at least 2 classes, found {num_classes}")

print("Classes:", base_dataset.classes)
print("Total images:", len(base_dataset))

train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size

train_idx, val_idx = random_split(
    range(len(base_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataset = Subset(
    datasets.ImageFolder(dataset_root, transform=train_transform),
    train_idx.indices
)

val_dataset = Subset(
    datasets.ImageFolder(dataset_root, transform=val_transform),
    val_idx.indices
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.features[-2:].parameters():
    param.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.35),
    nn.Linear(in_features, 256),
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

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

epochs = 8
best_val_acc = 0.0
best_model_path = "best_model.pth"

with mlflow.start_run() as run:
    mlflow.log_param("model", "efficientnet_b0")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 3e-4)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("dataset_root", str(dataset_root))
    mlflow.log_param("classes", ",".join(base_dataset.classes))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        mlflow.log_metrics(
            {
                "train_loss": train_loss / train_total,
                "train_accuracy": train_acc,
                "val_loss": val_loss / val_total,
                "val_accuracy": val_acc,
            },
            step=epoch,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    mlflow.log_metric("best_accuracy", best_val_acc)
    mlflow.log_artifact(best_model_path)

    with open("model_info.txt", "w", encoding="utf-8") as f:
        f.write(run.info.run_id)

    print(f"Run ID: {run.info.run_id}")
    print(f"Best Accuracy: {best_val_acc:.4f}")
