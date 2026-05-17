from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
DATA_DIR     = PROJECT_ROOT / "data"
LOGS_DIR     = PROJECT_ROOT / "logs"
MODELS_DIR   = PROJECT_ROOT / "models"
NOTEBOOKS_DIR= PROJECT_ROOT / "notebooks"
PLOTS_DIR    = PROJECT_ROOT / "plots"
RESULTS_DIR  = PROJECT_ROOT / "results"
SCRIPTS_DIR  = PROJECT_ROOT / "scripts"
TESTS_DIR    = PROJECT_ROOT / "tests"

for _d in [DATA_DIR, LOGS_DIR, MODELS_DIR, NOTEBOOKS_DIR, PLOTS_DIR, RESULTS_DIR, SCRIPTS_DIR, TESTS_DIR]:
    _d.mkdir(exist_ok=True)

ENV_FILE           = PROJECT_ROOT / ".env"
APP_ENTRYPOINT     = SRC_DIR / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Dataset téléchargé via Roboflow (emotion_scorer/download_dataset.py)
EMOTION_DATASET_DIR = PROJECT_ROOT / "emotion_scorer" / "Human-face-emotions-1"

# 8 classes d'émotions dans l'ordre des indices YOLO
EMOTION_CLASSES = ["anger", "content", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Poids pour le score humoriste (−1 → +1)
EMOTION_WEIGHTS = {
    "happy":    +1.0,
    "content":  +0.7,
    "surprise": +0.3,
    "neutral":   0.0,
    "sad":      -0.5,
    "fear":     -0.5,
    "anger":    -1.0,
    "disgust":  -1.0,
}

# Couleurs RGB pour l'UI (valeurs 0-255)
EMOTION_COLORS = {
    "happy":    (46,  204, 113),
    "content":  (39,  174, 96),
    "surprise": (241, 196, 15),
    "neutral":  (149, 165, 166),
    "sad":      (52,  152, 219),
    "fear":     (155, 89,  182),
    "anger":    (231, 76,  60),
    "disgust":  (192, 57,  43),
}

# Les 3 modèles à comparer.
# Place les fichiers .pt téléchargés depuis Kaggle dans models/ avec ces noms.
MODELS = {
    "yolov8n": {
        "name":        "YOLOv8n",
        "description": "YOLO Nano — ultra-rapide, idéal temps réel webcam",
        "type":        "yolo",
        "path":        MODELS_DIR / "yolov8n_emotions.pt",
    },
    "yolov8s": {
        "name":        "YOLOv8s",
        "description": "YOLO Small — meilleur compromis vitesse / précision",
        "type":        "yolo",
        "path":        MODELS_DIR / "yolov8s_emotions.pt",
    },
    "yolov8m": {
        "name":        "YOLOv8m",
        "description": "YOLO Medium — précision maximale",
        "type":        "yolo",
        "path":        MODELS_DIR / "yolov8m_emotions.pt",
    },
}
