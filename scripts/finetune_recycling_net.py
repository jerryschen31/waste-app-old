#!/usr/bin/env python3
"""
Fine-tune RecyclingNet (SigLIP) on the combined dataset.

This script fine-tunes a pre-trained SigLIP model to classify waste items
into 11 granular classes.
It uses Hugging Face Transformers library to ensure compatibility with
the Core ML export script.

Model: google/siglip-base-patch16-224
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
import json
import numpy as np
from tqdm import tqdm
from PIL import Image

# Configuration
DATA_DIR = Path("data_finetune")
OUTPUT_DIR = Path("outputs/finetune_recycling_net")
MODEL_ID = "google/siglip-base-patch16-224"
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Setup Data
    print("Preparing data transforms...")
    # SigLIP expects mean=0.5, std=0.5
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"

    if not train_dir.exists():
        print(f"Error: {train_dir} does not exist!")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    classes = train_dataset.classes
    num_classes = len(classes)
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in enumerate(classes)}
    
    print(f"Found {num_classes} classes: {classes}")

    # 2. Setup Model
    print(f"Loading model: {MODEL_ID}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    best_acc = 0.0
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save classes immediately
    with open(OUTPUT_DIR / "class_names.json", "w") as f:
        json.dump(classes, f)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            progress_bar.set_postfix({"loss": loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.logits.argmax(dim=-1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print("New best model! Saving...")
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            
    print(f"\nTraining complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
