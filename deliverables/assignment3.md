# Assignment 3 — Sélection et Évaluation des Modèles

## Projet : Comedian Emotion Scorer
**Objectif :** détecter les émotions faciales des spectateurs en temps réel et scorer un humoriste.

---

## 1. Définition du problème ML

Il s'agit d'un problème de **détection et classification multi-classes** en temps réel.

- **Entrée :** flux vidéo (webcam ou caméra de salle)
- **Sortie par frame :** liste de paires `(bounding_box, classe_émotion)` pour chaque visage détecté
- **Sortie finale :** score agrégé [0–100%] sur une fenêtre glissante de 150 frames
- **Classes :** 8 émotions — anger, content, disgust, fear, happy, neutral, sad, surprise
- **Contrainte forte :** inférence temps réel (≥ 15 fps), donc latence < 66 ms par frame

---

## 2. Définition de la métrique d'évaluation

### Métrique principale : **mAP50** (Mean Average Precision @ IoU 0.5)

La mAP50 est la métrique standard pour les modèles de détection d'objets. Elle mesure à la fois :
1. La **précision de la classification** (bonne classe prédite)
2. La **qualité de la localisation** (bounding box suffisamment proche du ground truth, IoU ≥ 0.5)

Elle est calculée pour chaque classe puis moyennée, ce qui est adapté à notre dataset quasi-équilibré.

### Métriques secondaires
| Métrique | Rôle |
|---|---|
| mAP50-95 | Précision à différents seuils IoU (robustesse) |
| F1-macro | Équilibre précision/rappel sur toutes les classes |
| Accuracy | Référence intuitive pour comparer avec ResNet/ViT |
| Inference time (ms/img) | Contrainte temps réel |
| Detection rate | % d'images avec au moins une détection |

### Justification du choix de la mAP50
- F1-score seul ignorerait la qualité des bounding boxes
- L'accuracy simple pénaliserait les modèles YOLO qui peuvent avoir 0 détection
- La mAP50 est le gold standard des benchmarks de détection (COCO, VOC)

---

## 3. Protocole d'évaluation

### Split du dataset
| Split | Images | Rôle |
|---|---|---|
| Train | 6 586 (70%) | Entraînement |
| Validation | 1 873 (20%) | Monitoring pendant l'entraînement (early stopping) |
| Test | 941 (10%) | Évaluation finale — **jamais vu pendant l'entraînement** |

Le split est fourni directement par Roboflow et est identique pour tous les modèles, garantissant une comparaison équitable.

### Protocole d'entraînement
- **Epochs :** 50 (avec early stopping patience=15)
- **Batch size :** 16
- **Image size :** 640 × 640
- **Optimiseur :** AdamW (détecté automatiquement par Ultralytics)
- **GPU :** Kaggle T4 (~2-3 min/epoch)

### Évaluation
Chaque modèle est évalué sur le **même jeu de test** via `scripts/main.py` :
```python
X_train, X_test, y_train, y_test = load_dataset_split()  # src/data.py
y_pred = model.predict(X_test)                            # src/model_io.py
metrics = compute_metrics(y_test, y_pred)                 # src/metrics.py
```

---

## 4. Présentation des trois modèles

---

### Modèle 1 — YOLOv8n (YOLO Nano)

**Architecture :** CSPDarknet + Path Aggregation Network (PAN) + Detection Head  
**Paramètres :** 3,2 M  
**Taille du modèle :** ~6 MB

#### Hypothèses principales
- Un backbone très léger suffit à capturer les patterns faciaux caractéristiques des émotions
- La localisation simultanée (bbox) compense la perte de précision liée à la taille réduite
- Les émotions sont des patterns visuels saillants (sourire, froncement de sourcils) détectables même avec peu de paramètres

#### Avantages attendus
- Inférence ultra-rapide (~2 ms/image sur GPU, ~15 ms sur CPU)
- Faible empreinte mémoire — déployable sur Raspberry Pi ou mobile
- Pipeline unifié : détection + classification en un seul forward pass

#### Limites attendues
- mAP50 inférieure aux modèles plus larges, notamment sur les émotions subtiles (content vs neutral)
- Moins robuste aux petits visages (spectateurs éloignés)
- Peut confondre anger/disgust dont les expressions sont visuellement proches

#### Adéquation avec le problème
**Excellente pour le cas d'usage temps réel.** Un spectacle nécessite un traitement fluide du flux vidéo. YOLOv8n peut tourner à 30+ fps sur GPU, rendant le feedback instantané. La légère perte de précision est acceptable car le score est une **moyenne sur 150 frames** — les erreurs ponctuelles sont lissées.

---

### Modèle 2 — YOLOv8s (YOLO Small)

**Architecture :** CSPDarknet (plus large) + PAN + Detection Head  
**Paramètres :** 11,2 M  
**Taille du modèle :** ~22 MB

#### Hypothèses principales
- Des couches plus larges permettent de capturer des représentations plus riches des micro-expressions
- Le compromis vitesse/précision de YOLOv8s est optimal pour une installation en salle de spectacle (GPU dédié disponible)
- 11 M de paramètres restent suffisamment légers pour une inférence rapide

#### Avantages attendus
- +5–8% mAP50 par rapport à YOLOv8n
- Meilleure discrimination des émotions proches (happy vs content, anger vs disgust)
- Robuste aux conditions d'éclairage difficiles grâce aux représentations plus profondes

#### Limites attendues
- Environ 2× plus lent que YOLOv8n (~4 ms/image)
- Nécessite un GPU pour maintenir 30 fps — moins adapté au déploiement CPU

#### Adéquation avec le problème
**Modèle recommandé pour la production.** C'est le meilleur équilibre entre précision métier (score fiable) et performance temps réel sur un GPU standard (RTX 3060 ou équivalent). C'est ce modèle qui sera proposé par défaut dans l'application Streamlit.

---

### Modèle 3 — YOLOv8m (YOLO Medium)

**Architecture :** CSPDarknet (large) + PAN + Detection Head  
**Paramètres :** 25,9 M  
**Taille du modèle :** ~52 MB

#### Hypothèses principales
- Une capacité représentationnelle plus élevée permet de mieux distinguer les 8 classes, notamment les émotions à faible variance visuelle
- La précision maximale est prioritaire dans un contexte d'analyse a posteriori d'un spectacle enregistré

#### Avantages attendus
- mAP50 la plus élevée des 3 modèles (~+3–5% vs YOLOv8s)
- Meilleure précision sur les visages partiellement occultés ou de profil
- Résultats les plus fiables pour l'analyse frame-by-frame d'une vidéo enregistrée

#### Limites attendues
- Plus lent (~8 ms/image) — peut nécessiter un GPU puissant pour 30 fps
- Risque de surapprentissage légèrement plus élevé sur un dataset de 9 400 images
- Fichier modèle plus lourd (52 MB)

#### Adéquation avec le problème
**Idéal pour l'analyse post-spectacle.** Si l'humoriste veut analyser un spectacle enregistré (non temps réel), YOLOv8m fournit les prédictions les plus précises pour identifier les moments exacts de peak d'hilarité.

---

## 5. Justification du choix des trois modèles

Les trois modèles sélectionnés sont tous des **variantes de YOLOv8** car :

1. **Pipeline unifié** : YOLO effectue détection + classification en un seul forward pass, ce qui est fondamental pour le temps réel. Les alternatives (ResNet-50 ou ViT seuls) nécessitent un détecteur de visages séparé.

2. **Comparaison contrôlée** : en comparant Nano vs Small vs Medium, on isole l'effet de la **capacité du modèle** à architecture constante. Cela permet une conclusion claire sur le trade-off vitesse/précision pour ce cas d'usage spécifique.

3. **Adéquation avec le dataset YOLO** : le dataset est annoté en format YOLOv8 (bounding boxes + classes). Utiliser YOLO tire parti des annotations de localisation que d'autres architectures ignoreraient.

4. **État de l'art en détection** : YOLOv8 est l'architecture de référence pour la détection temps réel en 2024–2025, avec un écosystème actif (Ultralytics) et une intégration native avec Roboflow.

### Pourquoi ResNet-50 et ViT ne sont pas les modèles principaux

Bien que ResNet-50 et ViT-Base aient été entraînés comme expériences complémentaires (notebooks Kaggle), ils ne constituent pas les modèles principaux pour les raisons suivantes :

- **Pas de localisation** : ils classifient l'image entière, pas les visages individuels
- **Pipeline plus complexe** : nécessitent un pre-processing de crop des visages
- **Latence plus élevée** : ViT-Base ~15–25 ms/image, incompatible avec 30 fps sans GPU dédié
- **Non adapté aux images multi-visages** : une salle contient des dizaines de spectateurs par frame

---

## 6. Notebooks et reproduction

### Entraînement des modèles YOLO
**Fichier :** `emotion_scorer/kaggle_train_emotions.ipynb`  
**Plateforme :** Kaggle (GPU T4 gratuit)  
**Exécution :**
1. Uploader le notebook sur kaggle.com/code
2. Activer GPU T4 (Settings → Accelerator)
3. Ajouter le secret `ROBOFLOW_API_KEY` dans Kaggle Secrets
4. Cliquer "Run All"
5. Télécharger `emotion_scorer_results.zip` depuis le panneau Files
6. Extraire les fichiers `.pt` dans `models/` :
   - `YOLOv8n_best.pt` → `models/yolov8n_emotions.pt`
   - `YOLOv8s_best.pt` → `models/yolov8s_emotions.pt`
   - `YOLOv8m_best.pt` → `models/yolov8m_emotions.pt`

### Évaluation comparative
```bash
# Une fois les modèles placés dans models/
python scripts/main.py
# → Génère results/model_metrics.csv
# → Lance l'app Streamlit sur http://localhost:8501
```

### App Streamlit standalone
```bash
cd ml-poc-project
PYTHONPATH=src streamlit run src/app.py
```
