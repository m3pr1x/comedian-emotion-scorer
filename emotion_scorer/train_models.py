"""
Script d'entraînement de plusieurs modèles de détection d'émotions faciales
Modèles testés: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv11n
Dataset: Human Face Emotions (9400 images, 8 classes)
"""

import os
import json
import time
import torch
from pathlib import Path
from ultralytics import YOLO


def detect_device() -> str:
    """Détecte le meilleur device disponible: MPS (Apple Silicon) > CUDA > CPU"""
    if torch.backends.mps.is_available():
        print("✅ GPU Apple Silicon (MPS) détecté — entraînement accéléré")
        return "mps"
    if torch.cuda.is_available():
        print(f"✅ GPU CUDA détecté: {torch.cuda.get_device_name(0)}")
        return "0"
    print("⚠️  Pas de GPU détecté — entraînement sur CPU (plus lent)")
    return "cpu"

# Configuration des modèles à tester
MODELS_TO_TEST = [
    {
        "name": "YOLOv8n",
        "model_path": "yolov8n.pt",
        "description": "YOLOv8 Nano - Ultra-léger, rapide"
    },
    {
        "name": "YOLOv8s", 
        "model_path": "yolov8s.pt",
        "description": "YOLOv8 Small - Bon équilibre vitesse/précision"
    },
    {
        "name": "YOLOv8m",
        "model_path": "yolov8m.pt",
        "description": "YOLOv8 Medium - Meilleure précision"
    },
    {
        "name": "YOLOv11n",
        "model_path": "yolo11n.pt",
        "description": "YOLO11 Nano - Dernière génération"
    },
]

# Chemin vers le dataset (après download)
# Modifier selon l'emplacement réel du dataset téléchargé
DATASET_YAML = "Human-face-emotions-1/data.yaml"

# Hyperparamètres d'entraînement (device détecté automatiquement au runtime)
TRAINING_CONFIG = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "patience": 15,
    "workers": 0,       # 0 requis pour MPS/CPU sur macOS (évite les deadlocks)
    "project": "runs/emotion_detection",
    "exist_ok": True,
}


def train_model(model_config: dict, dataset_yaml: str, device: str = "cpu") -> dict:
    """
    Entraîne un modèle YOLO et retourne les métriques.
    
    Args:
        model_config: Configuration du modèle (name, model_path, description)
        dataset_yaml: Chemin vers le fichier data.yaml du dataset
    
    Returns:
        dict avec les métriques de performance
    """
    model_name = model_config["name"]
    print(f"\n{'='*60}")
    print(f"Entraînement de {model_name}: {model_config['description']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Charger le modèle pré-entraîné
    model = YOLO(model_config["model_path"])
    
    # Entraîner le modèle
    results = model.train(
        data=dataset_yaml,
        name=model_name,
        device=device,
        **TRAINING_CONFIG
    )
    
    training_time = time.time() - start_time
    
    # Récupérer les métriques
    metrics = {
        "model_name": model_name,
        "description": model_config["description"],
        "training_time_seconds": round(training_time, 2),
        "training_time_formatted": f"{training_time/3600:.1f}h {(training_time%3600)/60:.0f}m",
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        "best_model_path": str(Path(TRAINING_CONFIG["project"]) / model_name / "weights/best.pt"),
    }
    
    print(f"\n✅ {model_name} terminé!")
    print(f"   mAP50: {metrics['mAP50']:.4f}")
    print(f"   mAP50-95: {metrics['mAP50_95']:.4f}")
    print(f"   Précision: {metrics['precision']:.4f}")
    print(f"   Rappel: {metrics['recall']:.4f}")
    print(f"   Temps: {metrics['training_time_formatted']}")
    
    return metrics


def train_all_models(dataset_yaml: str = DATASET_YAML, device: str = None) -> list:
    """
    Entraîne tous les modèles configurés et retourne les résultats.
    
    Args:
        dataset_yaml: Chemin vers le data.yaml
    
    Returns:
        Liste de dictionnaires avec les métriques de chaque modèle
    """
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(
            f"Dataset non trouvé: {dataset_yaml}\n"
            "Exécutez d'abord: python download_dataset.py"
        )
    
    all_results = []
    
    for model_config in MODELS_TO_TEST:
        try:
            metrics = train_model(model_config, dataset_yaml, device=device)
            all_results.append(metrics)
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement de {model_config['name']}: {e}")
            all_results.append({
                "model_name": model_config["name"],
                "error": str(e)
            })
    
    # Sauvegarder les résultats
    results_path = "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Résultats sauvegardés dans: {results_path}")
    return all_results


if __name__ == "__main__":
    device = detect_device()
    results = train_all_models(device=device)
    print("\n🎯 Entraînement de tous les modèles terminé!")
    print("Lancez compare_models.py pour voir la comparaison.")
