"""Métriques d'évaluation pour la détection d'émotions faciales.

compute_metrics() est appelé par scripts/main.py pour chaque modèle.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calcule accuracy, F1, précision et rappel.

    Les prédictions à -1 (aucune détection YOLO) sont comptées comme
    des erreurs dans le taux de détection mais exclues du calcul F1/accuracy
    pour ne pas pénaliser injustement les modèles de détection vs classification.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    detection_rate = float((y_pred != -1).mean())

    # Filtrer les non-détections pour les métriques de classification
    mask = y_pred != -1
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    if len(y_true_valid) == 0:
        return {
            "accuracy":        0.0,
            "f1_macro":        0.0,
            "f1_weighted":     0.0,
            "precision_macro": 0.0,
            "recall_macro":    0.0,
            "detection_rate":  0.0,
        }

    return {
        "accuracy":        float(accuracy_score(y_true_valid, y_pred_valid)),
        "f1_macro":        float(f1_score(y_true_valid, y_pred_valid, average="macro",    zero_division=0)),
        "f1_weighted":     float(f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true_valid, y_pred_valid, average="macro",    zero_division=0)),
        "detection_rate":  detection_rate,
    }
