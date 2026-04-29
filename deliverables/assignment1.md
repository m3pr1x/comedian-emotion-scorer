# Projet ML : Analyse de l'Hilarité du Public (Humorist Effectiveness Analysis)

## 1. Description du projet
Ce projet consiste à développer un système de **Vision par Ordinateur (Computer Vision)** capable d'évaluer en temps réel si un humoriste est "drôle" ou non en analysant les réactions faciales des spectateurs dans une salle de spectacle. L'idée est de transformer la perception subjective du rire en une donnée quantitative grâce à l'analyse d'images.

## 2. Définition du problème
Il s'agit d'un problème de **Classification Multi-classes** (ou binaire selon le raffinement). Le modèle doit classer les visages détectés dans différentes catégories d'émotions (ex: Joie/Rire, Neutre, Ennui/Tristesse). Le score global de "drôlerie" sera ensuite dérivé de l'agrégation de ces classifications sur l'ensemble de l'audience.

## 3. Description du dataset choisi
Le dataset utilisé provient de la plateforme **Roboflow**. Il est constitué d'images de visages annotées selon les expressions faciales. 
- **Source :** Roboflow (Dataset d'émotions faciales).
- **Format :** Images JPG/PNG avec annotations au format YOLO ou COCO.

## 4. Description des features disponibles
Les features extraites par le modèle seront :
- **Points d'ancrage faciaux (Landmarks) :** Position des yeux, de la bouche (étirement pour le rire), des sourcils.
- **Intensité des pixels :** Pour capturer les micro-expressions liées à l'amusement.
- **Bounding Boxes :** Localisation de chaque spectateur dans le champ de vision de la caméra.

## 5. Objectif Business
L'objectif est de fournir aux humoristes, producteurs de spectacles ou directeurs de salles un outil de **KPI (Key Performance Indicator)** pour :
- Mesurer l'efficacité d'un sketch minute par minute.
- Identifier les moments "mous" d'un spectacle.
- Comparer la réception d'un même spectacle selon les villes ou les publics.

## 6. Contexte Machine Learning
Le projet s'appuie sur le **Deep Learning**, plus particulièrement sur les **Réseaux de Neurones Convolutifs (CNN)** spécialisés dans la reconnaissance d'expressions faciales (FER - Facial Emotion Recognition). Des modèles pré-entraînés comme ResNet ou EfficientNet pourront être utilisés via Transfer Learning.

## 7. Métrique ou fonction de coût envisagée
- **Métrique principale :** F1-Score (pour équilibrer précision et rappel, car les moments de rire intense peuvent être plus rares que les moments neutres).
- **Fonction de coût :** Cross-Entropy Loss (classique pour la classification multi-classes).

## 8. Hypothèses, risques et limites identifiées
- **Conditions d'éclairage :** Les salles de spectacle sont souvent sombres, ce qui peut nuire à la qualité des images.
- **Orientation des visages :** Les spectateurs ne font pas tous face à la caméra.
- **Éthique et Vie Privée :** Le projet nécessite une réflexion sur l'anonymisation des données capturées (RGPD).
- **Biais du Dataset :** Le modèle doit être entraîné sur une diversité d'âges et d'ethnies pour être performant.

## 9. Données / Notebooks
### Obtention des données
Les données sont téléchargées via l'API Roboflow. Un script de téléchargement est fourni pour automatiser la récupération du dataset.

### Localisation dans le repository
- `/data/raw/` : Contient les images brutes (non versionnées si trop volumineuses).
- `/notebooks/01_EDA_Exploration.ipynb` : Analyse exploratoire (répartition des classes, exemples d'images).
- `/notebooks/02_Training_Model.ipynb` : Entraînement du modèle de classification.
POUR L'instant il n'y en a pas

### Exécution
1. Installer les dépendances : `pip install -r requirements.txt`.
2. Lancer le notebook d'EDA pour visualiser les données de Roboflow.
3. Pour le rendu final, exécuter le script `main.py` qui active la caméra et affiche les prédictions en temps réel.