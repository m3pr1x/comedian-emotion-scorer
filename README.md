# Comedian Emotion Scorer

Détecte les émotions faciales du public en temps réel et attribue un score à un humoriste.

**Pipeline :** Caméra → YOLO (détection + classification) → Score pondéré [0–100%]

## Démo rapide

```bash
git clone https://github.com/m3pr1x/ml-poc-project.git
cd ml-poc-project
pip install -r requirements.txt
python scripts/main.py
```

L'app Streamlit s'ouvre sur **http://localhost:8501**.

> Les modèles `.pt` ne sont pas inclus dans le repo (trop lourds). Voir la section **Modèles** ci-dessous.

---

## Structure du projet

```
ml-poc-project/
├── src/
│   ├── app.py          # App Streamlit (7 onglets)
│   ├── config.py       # Chemins, registre des modèles, config Streamlit
│   ├── data.py         # Chargement du dataset YOLO
│   ├── metrics.py      # Calcul des métriques (mAP50, F1, accuracy...)
│   └── model_io.py     # Wrapper YOLO sklearn-compatible
├── scripts/
│   └── main.py         # Point d'entrée unique : évaluation + lancement Streamlit
├── models/             # Dossier pour les fichiers .pt (non versionnés)
├── emotion_scorer/
│   ├── download_dataset.py          # Télécharge le dataset Roboflow
│   ├── kaggle_train_emotions.ipynb  # Entraînement YOLO sur Kaggle T4
│   └── training_results.json        # Résultats d'entraînement Kaggle
├── deliverables/       # Rendus académiques (assignment2.md, assignment3.md)
└── requirements.txt
```

---

## Modèles

Les modèles entraînés sur le dataset **Human Face Emotions** (Roboflow) ne sont pas inclus dans le repo car trop lourds (6-50 MB).

### Option A — Télécharger depuis Kaggle

1. Va sur le notebook Kaggle `kaggle-train-emotions`
2. Onglet **Output** → télécharge les fichiers `best.pt`
3. Place-les dans `models/` en les renommant :

```
models/
├── yolov8n_emotions.pt   (~6 MB  - Nano)
├── yolov8s_emotions.pt   (~22 MB - Small)
└── yolov8m_emotions.pt   (~50 MB - Medium)
```

### Option B — Re-entraîner en local

```bash
# 1. Télécharge le dataset (nécessite ROBOFLOW_API_KEY)
export ROBOFLOW_API_KEY=ta_cle_api
python emotion_scorer/download_dataset.py

# 2. Lance l'entraînement (~30 min sur GPU, ~3h sur CPU M2)
python emotion_scorer/train_models.py
```

---

## Dataset

**Source :** Human Face Emotions — Roboflow  
**9 400 images** annotées en format YOLOv8, réparties en **8 classes** :

`anger` · `content` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

**Split :** 70% train · 20% validation · 10% test

```bash
export ROBOFLOW_API_KEY=ta_cle_api
python emotion_scorer/download_dataset.py
```

---

## Lancer l'app seule (sans évaluation des modèles)

```bash
cd ml-poc-project
PYTHONPATH=src streamlit run src/app.py
```

---

## Résultats d'entraînement (Kaggle T4)

| Modèle  | mAP50  | Précision | Rappel | Temps |
|---------|--------|-----------|--------|-------|
| YOLOv8n | 0.7762 | 0.6814    | 0.7384 | 1h02  |
| YOLOv8s | 0.7767 | 0.6920    | 0.7154 | 1h41  |
| YOLOv8m | 0.7835 | 0.6605    | 0.7627 | 3h27  |

---

## Variables d'environnement

Crée un fichier `.env` à la racine du projet si besoin :

```env
ROBOFLOW_API_KEY=ta_cle_api
```

Le `.env` est exclu du repo par `.gitignore`.
