#!/usr/bin/env python3
"""Quick head-only fine-tune using PyTorch (1 epoch) on `data_balanced/`.

This is a fallback dry-run because TensorFlow isn't available in the venv.
It uses torchvision EfficientNet-B2 pretrained weights, freezes the backbone,
and trains a binary head for one epoch to validate the training pipeline.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import datasets, models


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data_balanced", help="Balanced dataset root")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", default="outputs/ft_quick_torch")
    p.add_argument("--unfreeze_blocks", type=int, default=0, help="Number of last feature blocks to unfreeze (partial unfreeze)")
    p.add_argument("--history", default="history.json", help="Filename to save training history in out_dir")
    return p.parse_args()


def build_model(device):
    # Load pretrained EfficientNet B2 from torchvision
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features if hasattr(model, 'classifier') else model.fc.in_features
    # EfficientNet in torchvision uses model.classifier = Sequential(Dropout, Linear)
    model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 1))

    model.to(device)
    return model


def partial_unfreeze(model, unfreeze_blocks: int):
    """Unfreeze the last `unfreeze_blocks` of model.features (if present)."""
    if unfreeze_blocks <= 0:
        return 0
    if not hasattr(model, "features"):
        return 0
    n = len(model.features)
    k = min(unfreeze_blocks, n)
    for i in range(n - k, n):
        for p in model.features[i].parameters():
            p.requires_grad = True
    return k


def get_dataloaders(data_dir, batch_size, num_workers=4, pin_memory=True):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    eval_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers // 2), pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers // 2), pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += images.size(0)
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    args = parse_args()
    # Prefer MPS on macOS (Apple Silicon) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    # Tune DataLoader parameters for target device
    if device.type == "mps":
        num_workers = 0
        pin_memory = False
    elif device.type == "cuda":
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 2
        pin_memory = False

    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, args.batch_size, num_workers=num_workers, pin_memory=pin_memory)
    model = build_model(device)

    # Partial unfreeze if requested
    unfrozen = partial_unfreeze(model, args.unfreeze_blocks)
    if unfrozen > 0:
        print(f"Unfroze last {unfrozen} feature blocks for training")

    criterion = nn.BCEWithLogitsLoss()
    # optimizer only for parameters with requires_grad=True
    params_to_opt = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_opt, lr=args.lr)

    print(f"Starting training on device={device}. Train batches: {len(train_loader)}")

    history = []
    best_val_acc = 0.0
    import time
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, time={epoch_time:.1f}s")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "time_s": epoch_time})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_model.pth"))

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    # save history
    import json
    hist_path = Path(args.out_dir) / args.history
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"Saved training history to {hist_path}")


if __name__ == "__main__":
    main()
