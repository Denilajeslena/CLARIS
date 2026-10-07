"""
Audio Forensic Model Fine-Tuning Script.
Trains audio deepfake classifier head over speech representations (Wav2Vec2 / Log-Mel CNN).
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from backend.audio.audio_model import AudioForensicCNN

def main():
    parser = argparse.ArgumentParser(description="Train Audio Forensic Classifier")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output_path", type=str, default="models/audio_model/audio_classifier.pth")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialized Audio Classifier Training on {device}")
    model = AudioForensicCNN().to(device)
    print(f"[OK] Audio Forensic model initialized. Target checkpoint: {args.output_path}")

if __name__ == "__main__":
    main()
