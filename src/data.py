"""Chargement du dataset Human Face Emotions (format YOLO).

load_dataset_split() est appelé par scripts/main.py pour évaluer les modèles.
X = liste de chemins d'images (str)
y = tableau numpy d'indices de classe (0-7)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _load_split(images_dir: Path, labels_dir: Path) -> tuple[list[str], np.ndarray]:
    """Charge les images et labels d'un split YOLO.

    Pour les images avec plusieurs annotations, prend la plus grande bbox
    (visage le plus proche de la caméra = dominant).
    """
    X, y = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in extensions:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        lines = [l.strip() for l in label_path.read_text().strip().splitlines() if l.strip()]
        if not lines:
            continue

        best_cls, best_area = None, -1.0
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            area = float(parts[3]) * float(parts[4])  # w * h (normalisé)
            if area > best_area:
                best_area = area
                best_cls = cls_id

        if best_cls is not None:
            X.append(str(img_path))
            y.append(best_cls)

    return X, np.array(y, dtype=int)


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Retourne (X_train, X_test, y_train, y_test) depuis le dataset YOLO."""
    from config import EMOTION_DATASET_DIR

    if not EMOTION_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {EMOTION_DATASET_DIR}\n"
            "Lance d'abord : python emotion_scorer/download_dataset.py"
        )

    train_imgs = EMOTION_DATASET_DIR / "train" / "images"
    train_lbls = EMOTION_DATASET_DIR / "train" / "labels"
    test_imgs  = EMOTION_DATASET_DIR / "test"  / "images"
    test_lbls  = EMOTION_DATASET_DIR / "test"  / "labels"

    X_train, y_train = _load_split(train_imgs, train_lbls)
    X_test,  y_test  = _load_split(test_imgs,  test_lbls)

    return X_train, X_test, y_train, y_test


def load_class_distribution(split: str = "train") -> dict[str, int]:
    """Retourne le nombre d'images par classe pour un split donné (app Streamlit)."""
    from config import EMOTION_DATASET_DIR, EMOTION_CLASSES

    lbls_dir = EMOTION_DATASET_DIR / split / "labels"
    if not lbls_dir.exists():
        return {}

    counts = {cls: 0 for cls in EMOTION_CLASSES}
    for label_file in lbls_dir.glob("*.txt"):
        lines = [l.strip() for l in label_file.read_text().strip().splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            if parts:
                cls_id = int(parts[0])
                if 0 <= cls_id < len(EMOTION_CLASSES):
                    counts[EMOTION_CLASSES[cls_id]] += 1

    return counts


def get_sample_image_paths(n_per_class: int = 1) -> dict[str, list[str]]:
    """Retourne n chemins d'images par classe pour la visualisation (app Streamlit)."""
    from config import EMOTION_DATASET_DIR, EMOTION_CLASSES

    train_imgs = EMOTION_DATASET_DIR / "train" / "images"
    train_lbls = EMOTION_DATASET_DIR / "train" / "labels"

    if not train_imgs.exists():
        return {}

    samples: dict[str, list[str]] = {cls: [] for cls in EMOTION_CLASSES}

    for img_path in sorted(train_imgs.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = train_lbls / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue
        cls_id = int(lines[0].split()[0])
        if 0 <= cls_id < len(EMOTION_CLASSES):
            cls_name = EMOTION_CLASSES[cls_id]
            if len(samples[cls_name]) < n_per_class:
                samples[cls_name].append(str(img_path))

        if all(len(v) >= n_per_class for v in samples.values()):
            break

    return samples
