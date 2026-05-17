# Assignment 2 — Préparation des Données & Feature Engineering

## Projet : Comedian Emotion Scorer
**Objectif :** détecter les émotions faciales des spectateurs et attribuer un score en temps réel à un humoriste.

---

## 1. Description des étapes de nettoyage des données

### Dataset source
Le dataset **Human Face Emotions** provient de Roboflow (workspace `louiss-workspace-jmpoo`, project `human-face-emotions-avush`, version 1). Il contient 9 400 images annotées en format YOLOv8 (fichiers `.txt` + `data.yaml`), réparties en 8 classes d'émotions.

### Nettoyage appliqué

**Vérification de l'intégrité des annotations**  
Lors du chargement dans `src/data.py`, chaque fichier label est vérifié :
- les fichiers `.txt` vides ou malformés (moins de 5 champs par ligne) sont ignorés
- les images sans fichier label correspondant sont exclues

**Gestion des images multi-visages**  
Certaines images contiennent plusieurs annotations (plusieurs visages). Pour garantir un label unique par image lors de l'évaluation comparative des modèles, on sélectionne la **bounding box de plus grande surface** (w × h normalisé), qui correspond au visage le plus proche de la caméra et donc le plus représentatif.

**Aucune suppression de classes**  
La distribution est quasi-équilibrée (~1 150–1 245 images par classe), aucun rééchantillonnage n'a été nécessaire.

---

## 2. Description des transformations appliquées

### Pour les modèles YOLO (détection)
Les modèles YOLO traitent les images en format BGR via OpenCV. Les transformations sont appliquées automatiquement par le pipeline Ultralytics :

| Transformation | Valeur | Justification |
|---|---|---|
| Redimensionnement | 640 × 640 px | Standard YOLOv8 — compromis précision/vitesse |
| Normalisation pixels | [0, 1] (÷255) | Requis par le backbone |
| Mosaic augmentation | activée | Améliore la robustesse aux petits objets |
| Random Flip horizontal | 50% | Invariance gauche/droite des visages |
| ColorJitter | auto | Robustesse aux variations d'éclairage (salles sombres) |

### Pour les modèles de classification (ResNet-50, ViT — notebooks Kaggle)
| Transformation | Train | Val/Test |
|---|---|---|
| Resize | 256 × 256 puis RandomCrop 224 | Resize 224 × 224 |
| Flip horizontal | 50% | Non |
| ColorJitter | brightness/contrast/saturation | Non |
| Normalisation | Mean=[0.485,0.456,0.406], Std=[0.229,0.224,0.225] (ImageNet) | Identique |

La normalisation ImageNet est imposée par les poids pré-entraînés : toute autre normalisation dégraderait les représentations apprises.

---

## 3. Description des nouvelles features créées

### Feature 1 — Score pondéré par émotion
Une nouvelle feature dérivée est calculée à partir des prédictions du modèle : le **Humor Score**. Ce n'est pas une feature d'entrée du modèle, mais une feature de sortie métier.

```python
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

# Score par frame :
frame_score = mean([EMOTION_WEIGHTS[e] for e in detected_emotions])

# Score final normalisé [0%, 100%] :
humor_score_pct = (rolling_mean(frame_scores, window=150) + 1) / 2 * 100
```

### Feature 2 — Fenêtre glissante temporelle
Pour lisser les variations frame-à-frame, un buffer circulaire de **150 frames** (~5 secondes à 30 fps) maintient l'historique des scores. Cette feature temporelle rend le score stable et représentatif d'une tendance plutôt que d'un instant ponctuel.

### Feature 3 — Taux de détection
Le ratio `detections / frames` est calculé comme indicateur secondaire : un public difficile à détecter (mauvais éclairage, visages de profil) est distingué d'un public effectivement peu expressif.

---

## 4. Justification des choix effectués

**Format YOLOv8 (détection vs classification)**  
Le format YOLO a été choisi car l'application finale filme une salle entière : il faut **localiser** chaque visage (bounding box) ET **classifier** son émotion simultanément. Un modèle de classification pure nécessiterait un détecteur de visages séparé, ajoutant de la latence.

**Poids des émotions**  
Les poids sont asymétriques : les émotions négatives (−1.0) ont un impact plus fort que les positives (+1.0) car un spectateur ennuyé ou dégoûté est un signal fort d'échec, alors que `surprise` peut être ambigüe. Ces valeurs s'inspirent des recherches en psychologie du rire (Ekman, 1999).

**Fenêtre de 150 frames**  
À 30 fps, 150 frames = 5 secondes, ce qui correspond au délai minimal pour qu'un effet comique soit "ressenti" par l'audience. Une fenêtre trop courte produit un score instable ; trop longue, elle masque les variations dynamiques du spectacle.

---

## 5. Alternatives testées et non retenues

### Alternative 1 — Modèle de classification pure (ResNet-50, ViT)
**Testée dans** : `kaggle_train_resnet50.ipynb`, `kaggle_train_vit.ipynb`  
**Principe** : classifier l'émotion de l'image entière sans localisation.  
**Rejet** : nécessite un détecteur de visages en amont (MTCNN ou RetinaFace), ce qui double la complexité du pipeline et augmente la latence. De plus, la classification globale d'image est moins précise quand plusieurs visages sont présents.

### Alternative 2 — Réduction de résolution à 416 × 416
**Considérée** pour réduire le temps d'entraînement sur CPU/M2.  
**Rejet** : une résolution plus faible réduit la précision de détection des petits visages (spectateurs au fond de la salle). 640 × 640 est maintenu pour la qualité.

### Alternative 3 — Score binaire (drôle / pas drôle)
**Considérée** pour simplifier l'interprétation.  
**Rejet** : perd la nuance des 8 émotions. La granularité permet de distinguer un public "content" d'un public "joyeux", ou un public "ennuyé" d'un public "dégoûté", ce qui est précieux pour l'analyse détaillée d'un spectacle.

### Alternative 4 — Rééchantillonnage (oversampling)
**Considérée** car la distribution des classes n'est pas parfaitement uniforme.  
**Rejet** : les écarts sont mineurs (±8%) — un rééchantillonnage aurait risqué d'introduire des artefacts sans gain mesurable.

---

## 6. Impact attendu des transformations sur les modèles

| Transformation | Impact attendu |
|---|---|
| Augmentation ColorJitter | +2–4% mAP50 sur images sombres (conditions spectacle) |
| Mosaic YOLO | +3–5% mAP50 sur détection multi-visages |
| Normalisation ImageNet | Accélère la convergence de ResNet/ViT × 3–5× |
| Fenêtre glissante 150 frames | Réduit la variance du score final de ~30% |
| Sélection bbox dominante | Uniformise l'évaluation — comparable à une classification single-label |

---

## 7. Données / Notebooks

### Obtention du dataset transformé
Le dataset est téléchargé via l'API Roboflow avec le script `emotion_scorer/download_dataset.py`. Les transformations d'augmentation sont appliquées **à la volée** lors de l'entraînement (pas de dataset préprocessé stocké).

### Localisation dans le repository
```
emotion_scorer/
├── Human-face-emotions-1/     # Dataset brut (non versionné sur Git — trop lourd)
│   ├── train/images/ + labels/
│   ├── valid/images/ + labels/
│   └── test/images/  + labels/
├── download_dataset.py        # Script de téléchargement
└── kaggle_train_emotions.ipynb # Entraînement YOLO avec augmentations
```

### Chargement et utilisation
```python
# Télécharger le dataset
python emotion_scorer/download_dataset.py

# Charger dans le pipeline d'évaluation
from data import load_dataset_split
X_train, X_test, y_train, y_test = load_dataset_split()

# Visualiser la distribution
from data import load_class_distribution
dist = load_class_distribution("train")  # dict {emotion: count}
```
