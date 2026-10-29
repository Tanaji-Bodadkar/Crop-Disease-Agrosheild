"""Train and evaluate AgroShield's ResNet18 CNN on a labeled leaf image dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agroshield.cnn import IMAGE_SIZE, MODEL_VERSION, build_model, eval_transform, train_transform

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def locate_imagefolder_root(data_dir: Path) -> Path:
    """Accept either an ImageFolder root or its parent directory."""
    candidates = [data_dir]
    candidates.extend(p for p in data_dir.rglob("*") if p.is_dir() and len(p.parts) - len(data_dir.parts) <= 3)
    for candidate in candidates:
        class_dirs = [p for p in candidate.iterdir() if p.is_dir() and not p.name.startswith(".")]
        image_classes = [p for p in class_dirs if any(x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES for x in p.iterdir())]
        if len(image_classes) >= 2:
            return candidate
    raise ValueError(f"No ImageFolder dataset found under {data_dir}. Expected one folder per class, containing images.")


def class_stratified_split(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[label].append(idx)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label, indices in by_class.items():
        if len(indices) < 3:
            raise ValueError(f"Class index {label} has fewer than 3 images; need at least 3 for train/validation/test splits.")
        train_part, holdout = train_test_split(indices, test_size=0.30, random_state=seed, shuffle=True)
        val_part, test_part = train_test_split(holdout, test_size=0.50, random_state=seed, shuffle=True)
        train_idx.extend(train_part)
        val_idx.extend(val_part)
        test_idx.extend(test_part)
    random.Random(seed).shuffle(train_idx)
    random.Random(seed + 1).shuffle(val_idx)
    random.Random(seed + 2).shuffle(test_idx)
    return train_idx, val_idx, test_idx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-per-class", type=int, default=250, help="Balanced per-class cap; set 0 to use every image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    torch.hub.set_dir(str(ROOT / ".cache" / "torch"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = locate_imagefolder_root(args.data_dir.resolve())
    discovery = ImageFolder(dataset_root)
    labels = discovery.classes
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(discovery.samples):
        by_class[label].append(idx)
    selected: list[int] = []
    rng = random.Random(args.seed)
    for label in range(len(labels)):
        indices = by_class[label]
        if args.max_per_class and len(indices) > args.max_per_class:
            indices = rng.sample(indices, args.max_per_class)
        selected.extend(indices)
    selected_targets = [discovery.targets[idx] for idx in selected]
    train_idx, val_idx, test_idx = class_stratified_split(selected_targets, args.seed)
    # Split indices are relative to the selected list; map back to ImageFolder indices.
    train_idx = [selected[i] for i in train_idx]
    val_idx = [selected[i] for i in val_idx]
    test_idx = [selected[i] for i in test_idx]

    train_data = ImageFolder(dataset_root, transform=train_transform())
    eval_data = ImageFolder(dataset_root, transform=eval_transform())
    workers = min(4, max(0, (torch.get_num_threads() // 2)))
    loader_options = {"batch_size": args.batch_size, "num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(train_data, train_idx), shuffle=True, **loader_options)
    val_loader = DataLoader(Subset(eval_data, val_idx), shuffle=False, **loader_options)
    test_loader = DataLoader(Subset(eval_data, test_idx), shuffle=False, **loader_options)

    print(f"Dataset: {dataset_root}")
    print(f"Images: {len(selected):,} | Classes: {len(labels)} | Train/validation/test: {len(train_idx):,}/{len(val_idx):,}/{len(test_idx):,}")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print("Loading ImageNet-pretrained ResNet18 weights (first run downloads about 45 MB)...")
    model = build_model(len(labels), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = -1.0
    best_state = None
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        count = 0
        for batch_idx, (images, targets) in enumerate(train_loader, 1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            count += targets.size(0)
            if batch_idx % 100 == 0 or batch_idx == len(train_loader):
                print(f"  epoch {epoch + 1}/{args.epochs} batch {batch_idx}/{len(train_loader)}", flush=True)
        scheduler.step()

        model.eval()
        val_correct = 0
        val_count = 0
        with torch.inference_mode():
            for images, targets in val_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                logits = model(images)
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_count += targets.size(0)
        val_accuracy = val_correct / max(val_count, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={running_loss / max(count, 1):.4f}, "
            f"train_acc={correct / max(count, 1):.4f}, val_acc={val_accuracy:.4f}",
            flush=True,
        )
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.to(device).eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    report = classification_report(y_true, y_pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
    metrics = {
        "model": MODEL_VERSION,
        "architecture": "ResNet18 CNN, initialized with ImageNet weights and fine-tuned on PlantVillage",
        "dataset_dir": str(dataset_root),
        "image_size": IMAGE_SIZE,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_validation_accuracy": best_val,
        "train_images": len(train_idx),
        "validation_images": len(val_idx),
        "test_images": len(test_idx),
        "n_images_used": len(selected),
        "n_classes": len(labels),
        "max_per_class": args.max_per_class,
        "class_counts_before_cap": {labels[i]: len(by_class[i]) for i in range(len(labels))},
        "class_counts_used": {
            labels[i]: min(len(by_class[i]), args.max_per_class) if args.max_per_class else len(by_class[i])
            for i in range(len(labels))
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "score_note": "Softmax scores are model scores and are not calibrated probabilities.",
        "field_performance_note": "PlantVillage holdout performance does not establish accuracy on field photographs.",
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "labels": labels, "model": MODEL_VERSION, "image_size": IMAGE_SIZE}, artifacts / "agroshield_cnn.pt")
    (artifacts / "cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (artifacts / "cnn_confusion_matrix.csv").write_text(
        "actual/predicted," + ",".join(labels) + "\n" + "".join(
            labels[row] + "," + ",".join(map(str, values)) + "\n"
            for row, values in enumerate(confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))))
        ),
        encoding="utf-8",
    )
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision/recall/F1: {metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}")
    print(f"Saved CNN model and evaluation to {artifacts}")
    print(f"Training and evaluation elapsed: {(time.time() - started) / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
