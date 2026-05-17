"""Chargement des modèles sérialisés.

Supporte : .pt (YOLO Ultralytics), .pth (PyTorch), .joblib, .pkl / .pickle
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class YOLOEmotionClassifier:
    """Wrapper sklearn-compatible autour d'un modèle YOLO Ultralytics.

    Expose predict(X) où X est une liste de chemins d'images.
    Retourne un tableau d'indices de classe (-1 = aucune détection).
    """

    def __init__(self, model_path: Path, conf: float = 0.30) -> None:
        from ultralytics import YOLO
        self.model = YOLO(str(model_path))
        self.conf  = conf

    def predict(self, X: list[str]) -> np.ndarray:
        preds = []
        for img_path in X:
            results = self.model(str(img_path), conf=self.conf, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                preds.append(-1)
                continue
            best_idx = int(boxes.conf.argmax())
            preds.append(int(boxes.cls[best_idx]))
        return np.array(preds, dtype=int)


def load_model(model_path: Path) -> Any:
    """Charge un modèle depuis le disque.

    - .pt   → YOLOEmotionClassifier (Ultralytics)
    - .joblib → joblib.load()
    - .pkl / .pickle → pickle.load()
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    suffix = model_path.suffix.lower()

    if suffix == ".pt":
        return YOLOEmotionClassifier(model_path)

    if suffix == ".joblib":
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Installe joblib : pip install joblib") from exc
        return joblib.load(model_path)

    if suffix in {".pkl", ".pickle"}:
        with model_path.open("rb") as fh:
            return pickle.load(fh)

    raise ValueError(f"Format non supporté : {model_path.suffix}. Utilise .pt, .joblib ou .pkl")
