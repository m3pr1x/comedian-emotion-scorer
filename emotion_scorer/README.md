# 🎤 Comedian Emotion Scorer

Détection des émotions faciales des spectateurs pour noter un humoriste en temps réel.

## 📊 Dataset

| Info | Valeur |
|------|--------|
| Source | Roboflow - Human Face Emotions |
| Images | 9 400 |
| Classes | 8 émotions |
| Split | 70% / 20% / 10% |
| Format | YOLOv8 (TXT + YAML) |

**Classes d'émotions:**
- 😠 anger (1 191 images)
- 😌 content (1 149 images)  
- 🤢 disgust (1 172 images)
- 😨 fear (1 184 images)
- 😄 happy (1 173 images)
- 😐 neutral (1 232 images)
- 😢 sad (1 192 images)
- 😲 surprise (1 245 images)

## 🏗️ Structure du Projet

```
emotion-scorer/
├── download_dataset.py     # Télécharge le dataset depuis Roboflow
├── train_models.py         # Entraîne plusieurs modèles YOLO
├── compare_models.py       # Compare les performances des modèles
├── predict_emotions.py     # Inférence en temps réel + scoring
├── requirements.txt        # Dépendances Python
├── .env.example            # Template de configuration
├── .gitignore              # Fichiers à ignorer
└── RESULTS.md              # Rapport généré après comparaison (auto-généré)
```

## 🚀 Démarrage Rapide

### 1. Installation

```bash
git clone https://github.com/votre-username/emotion-scorer
cd emotion-scorer
pip install -r requirements.txt
```

### 2. Configuration

```bash
cp .env.example .env
# Éditez .env et ajoutez votre clé API Roboflow
```

### 3. Télécharger le Dataset

```bash
python download_dataset.py
```

### 4. Entraîner les Modèles

```bash
python train_models.py
```

Modèles entraînés:
- **YOLOv8n** - Ultra-léger (vitesse prioritaire)
- **YOLOv8s** - Équilibre vitesse/précision
- **YOLOv8m** - Meilleure précision
- **YOLO11n** - Dernière génération

### 5. Comparer les Modèles

```bash
python compare_models.py
```

Génère: `RESULTS.md` et `model_comparison.png`

### 6. Scorer un Humoriste

```bash
# Avec la webcam
python predict_emotions.py --model runs/emotion_detection/YOLOv8s/weights/best.pt --source 0

# Avec une vidéo
python predict_emotions.py --model runs/emotion_detection/YOLOv8s/weights/best.pt --source video.mp4

# Sauvegarder le résultat
python predict_emotions.py --model ... --source 0 --output output.mp4
```

## 📈 Système de Score

| Émotion | Poids | Interprétation |
|---------|-------|----------------|
| 😄 happy | +1.0 | Très positif |
| 😌 content | +0.7 | Positif |
| 😲 surprise | +0.3 | Impact fort |
| 😐 neutral | 0.0 | Neutre |
| 😢 sad | -0.5 | Négatif |
| 😨 fear | -0.5 | Négatif |
| 😠 anger | -1.0 | Très négatif |
| 🤢 disgust | -1.0 | Très négatif |

**Score final** = Moyenne pondérée sur une fenêtre glissante de 150 frames  
Normalisé de **0%** (horrible) à **100%** (excellent)

## ⚙️ Configuration GPU

Modifiez `TRAINING_CONFIG` dans `train_models.py`:

```python
TRAINING_CONFIG = {
    "device": "0",   # GPU 0
    # "device": "cpu",  # CPU uniquement
    ...
}
```

## 📋 Résultats

Après entraînement, consultez `RESULTS.md` pour le rapport complet.

## 🔐 Sécurité

⚠️ **Ne jamais committer votre clé API Roboflow sur GitHub !**  
Le fichier `.env` est dans `.gitignore` pour cette raison.

## 📚 Technologies

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Détection d'objets
- [Roboflow](https://roboflow.com) - Dataset et gestion des annotations
- [OpenCV](https://opencv.org) - Traitement vidéo
