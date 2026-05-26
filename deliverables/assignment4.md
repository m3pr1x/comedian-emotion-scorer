# Assignment 4 — Visualisations et Analyse des Résultats

## Projet : Comedian Emotion Scorer
**Objectif :** détecter les émotions faciales des spectateurs en temps réel et scorer un humoriste de 0 à 100 %.

---

## 1. Visualisation des données brutes — Distribution des classes

### Objectif
Comprendre la répartition des 8 classes d'émotions dans le dataset avant tout entraînement.
Un dataset fortement déséquilibré biaiserait le modèle vers les classes surreprésentées.

### Type de graphique
**Bar chart** avec une barre par classe d'émotion, affichée pour chaque split (train / validation / test).

Chaque barre est colorée selon un code couleur propre à chaque émotion (repris dans toute l'application), ce qui facilite la lecture transversale entre les visualisations.

### Génération
La visualisation est générée dynamiquement dans l'application Streamlit :
- **Fichier :** `src/app.py` → fonction `_tab_dataset()`
- **Données sources :** annotations YOLO dans `emotion_scorer/Human-face-emotions-1/{train,valid,test}/labels/*.txt`
- **Script de chargement :** `src/data.py` → fonction `load_class_distribution(split)`

```bash
# Lancer l'app pour voir la visualisation interactive
cd ml-poc-project
PYTHONPATH=src python3 -m streamlit run src/app.py
# → Onglet "Dataset" → sous-onglets Train / Validation / Test
```

### Interprétation
Le dataset est **relativement équilibré** sur les 8 classes, avec ~1 000 à 1 400 annotations par émotion.
Les légères surreprésentations de `happy` et `neutral` reflètent la réalité du monde réel (expressions de repos et de joie sont plus fréquentes dans les données photographiques).

Cet équilibre justifie l'utilisation du **mAP50** comme métrique principale plutôt que l'accuracy pondérée, et rend le **F1-macro** (moyenne non pondérée sur les classes) pertinent comme métrique secondaire.

### Pertinence pour le projet
Vérifier l'équilibre du dataset est essentiel avant d'interpréter les résultats. Un modèle qui atteint 78 % de mAP50 sur un dataset équilibré est réellement performant. Sur un dataset biaisé, ce chiffre serait trompeur.

---

## 2. Visualisation après feature engineering — Batches d'entraînement augmentés

### Objectif
Visualiser les transformations appliquées aux images pendant l'entraînement.
Le feature engineering en vision par ordinateur consiste principalement en **augmentations de données** : artificellement diversifier les exemples vus par le modèle pour le rendre plus robuste.

### Type de graphique
**Grille d'images** (16 images par batch) générée automatiquement par Ultralytics à chaque run d'entraînement.

Chaque image montre :
- L'image transformée (crop, flip, HSV jitter, mosaic)
- Les bounding boxes annotées en overlay avec la classe correspondante

### Génération
Les images sont générées automatiquement lors de l'entraînement YOLO :
- **Fichier :** `emotion_scorer/kaggle_train_emotions.ipynb` (exécuté sur Kaggle GPU T4)
- **Sortie :** `emotion_scorer/runs/detect/runs/emotion_detection/YOLOv8n/train_batch{0,1,2}.jpg`

Ces images sont affichées dans l'application Streamlit :
```bash
# → Onglet "Modèles" → Section "Exemples de batches d'entraînement"
```

### Augmentations appliquées

| Augmentation | Valeur | Effet |
|---|---|---|
| Mosaic | 100 % | Combine 4 images en une — enrichit les contextes |
| Flip horizontal | 50 % | Double la variété des orientations |
| HSV Saturation | ±70 % | Robustesse aux éclairages variables |
| HSV Value (brightness) | ±40 % | Gère les salles sombres / éclairées |
| Scale | ±50 % | Gère les visages proches et éloignés |
| Random Erasing | 40 % | Force le modèle à ne pas dépendre d'une seule zone du visage |

### Interprétation
La **mosaic augmentation** est la plus importante : elle combine 4 images annotées en une seule, ce qui permet au modèle de voir des visages à des échelles et dans des contextes très variés à chaque batch. Cela explique en grande partie pourquoi même le plus petit modèle (YOLOv8n, 3.2 M paramètres) atteint 77.6 % de mAP50.

### Pertinence pour le projet
Dans un contexte de comedy club, le modèle sera confronté à des conditions variées : éclairage de scène, visages partiellement masqués, spectateurs à différentes distances. Les augmentations simulaient ces conditions pendant l'entraînement, rendant le modèle plus robuste en production.

---

## 3. Visualisation des performances — Courbes d'entraînement, Matrice de confusion et Radar chart

Trois visualisations complémentaires sont utilisées pour évaluer et comparer les modèles.

---

### 3a. Courbes d'entraînement

#### Objectif
Suivre l'évolution des métriques epoch par epoch pour vérifier que le modèle apprend correctement et détecter un éventuel surapprentissage.

#### Type de graphique
**Courbes temporelles** (ligne) avec l'epoch en abscisse et la valeur de la métrique en ordonnée.
Les métriques suivies : `box_loss`, `cls_loss`, `mAP50`, `mAP50-95`.

#### Génération
Générées automatiquement par Ultralytics pendant l'entraînement :
- **Fichier source :** `emotion_scorer/assets/results_yolov8{n,s,m}.png`
- **Données brutes :** `runs/emotion_detection/YOLOv8{n,s,m}/results.csv`
- **Affichage :** `src/app.py` → Onglet "Résultats" → Section "Visualisations d'entraînement"

#### Interprétation
Les deux courbes de loss (`box_loss` et `cls_loss`) descendent régulièrement et se stabilisent sans remontée — **pas de surapprentissage détecté**.

Le mAP50 monte et converge autour de **78 %** sur les 3 modèles.
La convergence rapide (plateau atteint vers l'epoch 30–35) confirme l'efficacité du **transfer learning** depuis les poids COCO pré-entraînés.

---

### 3b. Matrice de confusion normalisée

#### Objectif
Identifier quelles émotions sont bien reconnues et lesquelles sont confondues entre elles.
La matrice de confusion révèle les **patterns d'erreur** du modèle, ce que le mAP50 global masque.

#### Type de graphique
**Heatmap normalisée** (valeurs entre 0 et 1) :
- Lignes = classe réelle
- Colonnes = classe prédite
- Diagonale = bonnes prédictions
- Hors-diagonale = confusions

#### Génération
Générée automatiquement par Ultralytics à la fin de l'entraînement :
- **Fichier source :** `emotion_scorer/assets/confusion_matrix_yolov8{n,s,m}.png`
- **Affichage :** `src/app.py` → Onglet "Résultats" → Section "Visualisations d'entraînement"

#### Interprétation
**Émotions bien reconnues :**
- `happy` — sourire large, yeux plissés : signal visuel fort et distinctif
- `anger` — sourcils froncés, mâchoire serrée : contraste fort avec les autres classes

**Principales confusions :**
- `content` ↔ `neutral` — un sourire léger et une expression neutre partagent des traits faciaux proches
- `fear` ↔ `surprise` — yeux écarquillés communs aux deux classes

Ces confusions sont **cohérentes avec la psychologie des émotions** : les paires confondues sont également difficiles à distinguer pour un humain sur une photo statique.

---

### 3c. Radar chart comparatif

#### Objectif
Comparer les 3 modèles simultanément sur plusieurs métriques d'un seul coup d'œil.
Le radar chart permet une lecture globale là où un tableau de chiffres est moins intuitif.

#### Type de graphique
**Radar chart** (ou spider chart) avec 5 axes : accuracy, F1-macro, F1-weighted, précision, rappel.
Chaque modèle est représenté par un polygone coloré.

#### Génération
Généré dynamiquement en Python avec Plotly depuis le fichier de métriques :
- **Données sources :** `results/model_metrics.csv` (généré par `scripts/main.py`)
- **Fichier :** `src/app.py` → fonction `_tab_results()`
- **Affichage :** `src/app.py` → Onglet "Résultats" → "Radar chart"

```bash
# Générer model_metrics.csv
python3 scripts/main.py
# → Évalue les 3 modèles sur le jeu de test
# → Sauvegarde results/model_metrics.csv
# → Lance l'app Streamlit
```

#### Interprétation
Les 3 polygones sont quasi-superposés — les 3 variantes YOLOv8 convergent vers des performances similaires sur ce dataset.

L'écart maximal entre YOLOv8n et YOLOv8m est de **+0.7 % de mAP50** (77.6 % → 78.3 %), ce qui est faible.

Cela s'explique par la nature de la tâche : les émotions faciales sont des **patterns visuels saillants** (sourire, sourcils froncés) que même un réseau léger capture efficacement. Le goulot d'étranglement n'est pas la capacité du modèle mais la **variété du dataset** (conditions d'éclairage, angles de tête).

---

## 4. Justification globale de la pertinence des visualisations

Les trois niveaux de visualisation couvrent l'ensemble du cycle ML :

| Étape | Visualisation | Question répondue |
|---|---|---|
| Données brutes | Distribution des classes | Le dataset est-il équilibré ? |
| Feature engineering | Batches augmentés | Le modèle est-il exposé à des données suffisamment variées ? |
| Performances | Courbes + Matrice + Radar | Le modèle a-t-il appris ? Quelles sont ses erreurs ? Quel modèle choisir ? |

Ensemble, elles permettent de construire une **décision éclairée** :

- Le dataset équilibré → le mAP50 est une métrique fiable
- Les augmentations agressives → expliquent la robustesse en conditions réelles
- La matrice de confusion → justifie le choix du seuil de confiance à 0.25 dans la démo
- Le radar chart → confirme que YOLOv8n est suffisant pour le temps réel (–0.7 % de mAP pour 4× plus de vitesse)

---

## 5. Localisation des fichiers et reproduction

```
ml-poc-project/
├── src/
│   ├── app.py              # Toutes les visualisations interactives (Streamlit)
│   └── data.py             # Chargement dataset + distribution des classes
├── scripts/
│   └── main.py             # Évaluation des modèles → results/model_metrics.csv
├── emotion_scorer/
│   ├── kaggle_train_emotions.ipynb   # Entraînement + génération des courbes
│   ├── assets/
│   │   ├── results_yolov8{n,s,m}.png              # Courbes d'entraînement
│   │   ├── confusion_matrix_yolov8{n,s,m}.png     # Matrices de confusion
│   │   └── pr_curve_yolov8{n,s,m}.png             # Courbes Précision/Rappel
│   └── runs/detect/runs/emotion_detection/YOLOv8n/
│       └── train_batch{0,1,2}.jpg   # Batches d'entraînement augmentés
└── results/
    └── model_metrics.csv    # Métriques d'évaluation finale
```

### Commandes de reproduction

```bash
# 1. Télécharger le dataset
export ROBOFLOW_API_KEY=ta_cle_api
python emotion_scorer/download_dataset.py

# 2. Évaluer les modèles et lancer l'app
cd ml-poc-project
PYTHONPATH=src python3 scripts/main.py

# 3. Ou lancer l'app seule (sans évaluation)
PYTHONPATH=src python3 -m streamlit run src/app.py
```
