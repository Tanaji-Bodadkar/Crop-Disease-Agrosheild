"""Train and evaluate the AgroShield image-classification baseline."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agroshield.features import FEATURE_VERSION, IMAGE_SIZE, extract_features, preprocess

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def class_folders(root: Path) -> list[tuple[str, list[Path]]]:
    """Find class folders containing images directly or under an images folder."""
    found: dict[str, list[Path]] = {}
    for folder in root.rglob("*"):
        if not folder.is_dir() or any(part.startswith(".") or part == "__MACOSX" for part in folder.parts):
            continue
        files = [x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES]
        if not files:
            continue
        label = folder.name
        if label.casefold() == "images":
            label = folder.parent.name
        found.setdefault(label, []).extend(files)
    return sorted((name, paths) for name, paths in found.items() if len(paths) >= 2)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        return preprocess(rgb)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw", help="Folder containing one subfolder per class")
    parser.add_argument("--max-per-class", type=int, default=500, help="Maximum images used from each class; set 0 to use all")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction held out for final evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Dataset folder not found: {args.data_dir}\nRun: python scripts/download_plantvillage.py", file=sys.stderr)
        return 2
    folders = class_folders(args.data_dir)
    if len(folders) < 2:
        print("Could not find at least two class folders with image files. Pass the folder containing the disease-class directories with --data-dir.", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    records: list[tuple[Path, str]] = []
    class_counts = {}
    for label, paths in folders:
        paths = sorted(paths)
        if args.max_per_class and len(paths) > args.max_per_class:
            paths = sorted(rng.sample(paths, args.max_per_class))
        class_counts[label] = len(paths)
        records.extend((path, label) for path in paths)
    if min(class_counts.values()) < 2:
        print("Each class needs at least two images for a stratified train/test split.", file=sys.stderr)
        return 2

    paths = np.asarray([str(path) for path, _ in records], dtype=object)
    labels = np.asarray([label for _, label in records], dtype=object)
    train_paths, test_paths, y_train, y_test = train_test_split(
        paths, labels, test_size=args.test_size, random_state=args.seed, stratify=labels
    )
    features = []
    kept_labels = []
    kept_paths = []
    total = len(records)
    started = time.time()
    for idx, path_text in enumerate(paths, 1):
        try:
            feat = extract_features(read_rgb(Path(path_text)))
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            print(f"Skipping unreadable image {path_text}: {exc}", file=sys.stderr)
            continue
        features.append(feat)
        kept_labels.append(labels[idx - 1])
        kept_paths.append(path_text)
        if idx % 500 == 0 or idx == total:
            print(f"\rProcessed {idx}/{total} images", end="", flush=True)
    print(f"\nFeature extraction: {time.time() - started:.1f}s")
    if len(features) != total:
        path_to_feature = {p: f for p, f in zip(kept_paths, features)}
        usable_train = [i for i, x in enumerate(train_paths) if x in path_to_feature]
        usable_test = [i for i, x in enumerate(test_paths) if x in path_to_feature]
        X_train = np.stack([path_to_feature[train_paths[i]] for i in usable_train])
        X_test = np.stack([path_to_feature[test_paths[i]] for i in usable_test])
        y_train = y_train[usable_train]
        y_test = y_test[usable_test]
    else:
        matrix = np.stack(features)
        train_index = {p: i for i, p in enumerate(paths)}
        X_train = matrix[[train_index[p] for p in train_paths]]
        X_test = matrix[[train_index[p] for p in test_paths]]
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print("Too few readable examples remain for evaluation. Check the dataset folders.", file=sys.stderr)
        return 2

    model = RandomForestClassifier(
        n_estimators=120,
        max_features="sqrt",
        max_depth=28,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    print(f"Training on {len(X_train):,} images across {len(set(y_train))} classes...")
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    labels_order = sorted(set(labels))
    report = classification_report(y_test, predicted, labels=labels_order, output_dict=True, zero_division=0)
    metrics = {
        "dataset_dir": str(args.data_dir.resolve()),
        "feature_version": FEATURE_VERSION,
        "input_size": IMAGE_SIZE,
        "seed": args.seed,
        "test_fraction": args.test_size,
        "max_per_class": args.max_per_class,
        "n_images_used": len(features),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_classes": len(labels_order),
        "class_counts_before_read_errors": class_counts,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "score_note": "Random Forest class scores are uncalibrated and are not calibrated probabilities.",
        "severity": "Not modeled; dataset labels do not define severity.",
    }

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    artifact_path = artifacts / "agroshield_model.joblib"
    joblib.dump({"model": model, "labels": labels_order, "feature_version": FEATURE_VERSION, "input_size": IMAGE_SIZE}, artifact_path)
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_test, predicted, labels=labels_order)
    with (artifacts / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted", *labels_order])
        for label, row in zip(labels_order, cm):
            writer.writerow([label, *row.tolist()])
    print(f"Held-out accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro precision/recall/F1: {metrics['macro_precision']:.3f} / {metrics['macro_recall']:.3f} / {metrics['macro_f1']:.3f}")
    print(f"Saved model and evaluation to {artifacts}")
    print("These benchmark results do not establish field performance. Evaluate separately on field images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
