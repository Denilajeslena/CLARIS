"""
Forensic Classifier Training Script with Transfer Learning & Robustness Augmentations.
Supports backbone freezing, two-stage fine-tuning, and perturbation-resistant data augmentation.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import io
import numpy as np

# Custom Forensic Augmentations
class RandomJPEGCompression:
    def __init__(self, quality_min=30, quality_max=85, p=0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(self.quality_min, self.quality_max)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            return Image.open(buf)
        return img

class RandomDownscaleUpscale:
    def __init__(self, scale_min=0.4, scale_max=0.8, p=0.4):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            w, h = img.size
            scale = random.uniform(self.scale_min, self.scale_max)
            small = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            return small.resize((w, h), Image.Resampling.BICUBIC)
        return img

def get_transforms(image_size=256, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomJPEGCompression(quality_min=35, quality_max=85, p=0.5),
            RandomDownscaleUpscale(scale_min=0.5, scale_max=0.8, p=0.4),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)

def main():
    parser = argparse.ArgumentParser(description="Train Forensic CNN Classifier")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze backbone in initial phase")
    parser.add_argument("--output_path", type=str, default="models/forensic_model/best_checkpoint.pth")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} | Freeze Backbone: {args.freeze_backbone}")

if __name__ == "__main__":
    main()
