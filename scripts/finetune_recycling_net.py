#!/usr/bin/env python3
"""
Fine-tune RecyclingNet (SigLIP) on the combined dataset.

This script fine-tunes a pre-trained SigLIP model to classify waste items
into 11 granular classes, which map to 5 main categories.
1. e-Waste (electronics, batteries, lightbulbs)
2. Paper (paper, cardboard)
3. Composite (clothes, shoes)
4. Plastic (plastic)
5. Glass/Metal (glass, metal, trash)

The model is saved as a SafeTensors checkpoint or PyTorch .pt file.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pathlib import Path
import timm
import argparse
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import classification_report

# Configuration
DATA_DIR = Path("data_finetune")
OUTPUT_DIR = Path("outputs/finetune_recycling_net")
MODEL_NAME = "vit_so400m_patch14_siglip_224"  # Or "vit_base_patch16_224" depending on what we used
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # SigLIP normalization usually 0.5
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (total/loader.batch_size), 'acc': correct/total})
        
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return running_loss / len(loader), correct / total, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Fine-tune RecyclingNet")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    
    if not train_dir.exists():
        print(f"Error: Data directory {train_dir} does not exist.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(is_train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    
    # 2. Setup Model
    print(f"Creating model {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save class names mapping
    with open(OUTPUT_DIR / "classes.json", "w") as f:
        json.dump(classes, f)
        
    best_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print("New best accuracy! Saving model...")
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            
            # Save detailed report
            report = classification_report(val_labels, val_preds, target_names=classes, output_dict=True)
            with open(OUTPUT_DIR / f"val_report_epoch_{epoch+1}.json", "w") as f:
                json.dump(report, f, indent=2)
                
    print(f"Training complete. Best Validation Accuracy: {best_acc:.4f}")
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
