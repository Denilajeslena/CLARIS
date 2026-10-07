"""
One-shot pipeline: Download → Sample 100 real + 100 fake → Split 70/30 → Train → Evaluate.
Dataset: ciplab/real-and-fake-face-detection — pre-cropped face images.

Usage:
    venv/bin/python scripts/run_pipeline.py

Requires ~/.kaggle/kaggle.json with your Kaggle API token.
"""

import os, sys, random, shutil, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "mini_dataset"
KAGGLE_DIR  = ROOT / "data" / "kaggle_raw" / "faces"
MODEL_OUT   = ROOT / "models" / "forensic_model" / "best_checkpoint.pth"
TRAIN_LOG   = ROOT / "outputs" / "reports" / "train_log.json"
EVAL_REPORT = ROOT / "outputs" / "reports" / "mini_eval.json"

REAL_COUNT  = 500
FAKE_COUNT  = 500
TRAIN_RATIO = 0.70
SEED        = 42
EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 1e-4
IMAGE_SIZE  = 224

random.seed(SEED)
np.random.seed(SEED)

# ── Dataset class (module-level — required for Python 3.14 pickling) ───────────
class FaceDataset(Dataset):
    def __init__(self, split, transform):
        self.transform = transform
        real_dir = DATA_DIR / split / "real"
        fake_dir = DATA_DIR / split / "fake"
        self.samples = (
            [(p, 0) for p in real_dir.glob("*.jpg")] +
            [(p, 1) for p in fake_dir.glob("*.jpg")]
        )
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

# ── Step 1: Download ───────────────────────────────────────────────────────────
def download_dataset():
    if KAGGLE_DIR.exists() and any(KAGGLE_DIR.rglob("*.jpg")):
        print("[SKIP] Dataset already downloaded.")
        return
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    print("[1/4] Downloading ciplab/real-and-fake-face-detection from Kaggle...")
    import subprocess
    kaggle_bin = Path(sys.executable).parent / "kaggle"
    result = subprocess.run(
        [str(kaggle_bin), "datasets", "download", "-d", "ciplab/real-and-fake-face-detection",
         "--unzip", "-p", str(KAGGLE_DIR)],
        capture_output=False
    )
    if result.returncode != 0:
        print("\n[ERROR] Kaggle download failed.")
        sys.exit(1)
    print("[OK] Download complete.")

# ── Step 2: Sample & split ─────────────────────────────────────────────────────
def sample_and_split():
    if (DATA_DIR / "train" / "real").exists():
        count = len(list((DATA_DIR / "train" / "real").glob("*.jpg"))) + \
                len(list((DATA_DIR / "test"  / "real").glob("*.jpg")))
        if count >= REAL_COUNT:
            print("[SKIP] Mini dataset already prepared.")
            return

    print("[2/4] Sampling and splitting dataset...")

    # ciplab dataset has explicit training_real / training_fake folders
    real_dir = KAGGLE_DIR / "real_and_fake_face" / "training_real"
    fake_dir = KAGGLE_DIR / "real_and_fake_face" / "training_fake"

    if not real_dir.exists() or not fake_dir.exists():
        print(f"[ERROR] Expected folders not found: {real_dir} / {fake_dir}")
        sys.exit(1)

    real_imgs = list(real_dir.glob("*.jpg"))
    fake_imgs = list(fake_dir.glob("*.jpg"))
    print(f"  Found {len(real_imgs)} real, {len(fake_imgs)} fake images.")

    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)

    n_train = int(REAL_COUNT * TRAIN_RATIO)
    splits = {
        "train/real": real_imgs[:n_train],
        "train/fake": fake_imgs[:n_train],
        "test/real":  real_imgs[n_train:REAL_COUNT],
        "test/fake":  fake_imgs[n_train:FAKE_COUNT],
    }
    for split_path, imgs in splits.items():
        dest = DATA_DIR / split_path
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dest / img.name)

    print(f"  Train: {n_train} real + {n_train} fake")
    print(f"  Test:  {REAL_COUNT - n_train} real + {FAKE_COUNT - n_train} fake")
    print("  Cleaning up raw download to save disk space...")
    shutil.rmtree(KAGGLE_DIR, ignore_errors=True)
    print(f"  Freed: {KAGGLE_DIR}")
    print("[OK] Dataset prepared.")

# ── Step 3: Train ──────────────────────────────────────────────────────────────
def train():
    sys.path.insert(0, str(ROOT))
    from backend.vision.forensic_classifier import ForensicBackbone

    train_tf = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.15, 0.15, 0.15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = FaceDataset("train", train_tf)
    test_ds  = FaceDataset("test",  test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3/4] Training on {device} | {len(train_ds)} train, {len(test_ds)} test samples")

    model = ForensicBackbone(backbone_name="efficientnet_b4", pretrained=True, num_classes=2).to(device)
    for p in model.features.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    epoch_logs = []
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        if epoch == 4:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR * 0.1)
            print("  [Epoch 4] Backbone unfrozen for fine-tuning.")

        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)

        epoch_logs.append({"epoch": epoch, "loss": round(train_loss, 4),
                            "train_acc": round(train_acc, 4), "test_acc": round(val_acc, 4),
                            "best": is_best})
        with open(TRAIN_LOG, "w") as f:
            json.dump({"epochs": epoch_logs, "best_test_acc": round(best_acc, 4),
                       "model_saved_to": str(MODEL_OUT)}, f, indent=2)

        marker = " ← best" if is_best else ""
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | Test Acc: {val_acc:.3f}{marker}")

    print(f"[OK] Training complete. Best test accuracy: {best_acc:.3f}")
    print(f"     Model → {MODEL_OUT}")
    print(f"     Log   → {TRAIN_LOG}")

# ── Step 4: Evaluate ───────────────────────────────────────────────────────────
def evaluate():
    import torch.nn.functional as F
    from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                                  f1_score, precision_score, recall_score,
                                  confusion_matrix)
    sys.path.insert(0, str(ROOT))
    from backend.vision.forensic_classifier import ForensicBackbone

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForensicBackbone(backbone_name="efficientnet_b4", pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.eval()

    ds = FaceDataset("test", transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits, _ = model(imgs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "roc_auc":           round(float(roc_auc_score(y_true, y_prob)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_score":          round(float(f1_score(y_true, y_pred)), 4),
        "precision":         round(float(precision_score(y_true, y_pred)), 4),
        "recall_tpr":        round(float(recall_score(y_true, y_pred)), 4),
        "fpr":               round(float(fp / (fp + tn + 1e-8)), 4),
        "fnr":               round(float(fn / (fn + tp + 1e-8)), 4),
        "test_samples":      len(y_true),
        "confusion_matrix":  cm.tolist(),
    }

    print("\n[4/4] ══════════ EVALUATION RESULTS ══════════")
    for k, v in results.items():
        if k != "confusion_matrix":
            print(f"  {k:<22} {v}")
    print(f"\n  Confusion Matrix (TN FP / FN TP):")
    print(f"    {tn:4d}  {fp:4d}")
    print(f"    {fn:4d}  {tp:4d}")
    print("═" * 44)

    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_REPORT, "w") as f:
        json.dump(results, f, indent=2)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  OUTPUT FILES")
    print(f"  Trained model  → {MODEL_OUT}")
    print(f"  Training log   → {TRAIN_LOG}")
    print(f"  Eval report    → {EVAL_REPORT}")
    print("  Sampled images → data/mini_dataset/")
    print("    ├── train/real/  (70 images)")
    print("    ├── train/fake/  (70 images)")
    print("    ├── test/real/   (30 images)")
    print("    └── test/fake/   (30 images)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    download_dataset()
    sample_and_split()
    train()
    evaluate()
