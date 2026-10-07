"""
Dataset Preparation & Anti-Overfitting Preprocessing Script.
Supports FaceForensics++, DFDC, Celeb-DF, and DeeperForensics.
CRITICAL INTEGRITY PRINCIPLE: Video-identity-level splitting (zero frame leakage across splits).
"""

import os
import argparse
import random
import shutil
from typing import Dict, List

def create_directory_structure(base_dir: str):
    splits = ["train/real", "train/fake", "val/real", "val/fake", "test/real", "test/fake", "unseen"]
    for s in splits:
        os.makedirs(os.path.join(base_dir, s), exist_ok=True)
    print(f"[OK] Initialized dataset hierarchy at {base_dir}")

def split_by_video_identity(
    video_manifest: Dict[str, str], # {video_id: 'real' | 'fake'}
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Prevents temporal data leakage!
    Splits strictly by unique Subject/Video ID so that frames from the same source video
    CANNOT appear in both training and test sets.
    """
    random.seed(seed)
    ids = list(video_manifest.keys())
    random.shuffle(ids)

    n_train = int(len(ids) * train_ratio)
    n_val = int(len(ids) * val_ratio)

    splits = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:]
    }

    print(f"--- Anti-Data Leakage Subject Split ---")
    print(f"Total Unique Videos: {len(ids)}")
    print(f"Train IDs: {len(splits['train'])} | Val IDs: {len(splits['val'])} | Test IDs: {len(splits['test'])}")
    return splits

def main():
    parser = argparse.ArgumentParser(description="Dataset setup for Multimodal Forensics")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data folder")
    parser.add_argument("--dataset_type", type=str, default="ffpp", choices=["ffpp", "celeb_df", "dfdc", "custom"])
    args = parser.parse_args()

    create_directory_structure(args.data_dir)
    print("\n[NOTE] As per competition rules, large raw video datasets (FaceForensics++, Celeb-DF) are not auto-downloaded.")
    print("Please place extracted face crops or video files into the respective data/ folders.")

if __name__ == "__main__":
    main()
