# Prompt pour l'extension Claude sur Chrome

---

Tu es mon assistant pour un projet de computer vision. Je dois entraîner des modèles de détection d'émotions faciales sur Kaggle (GPU gratuit dans le cloud). Guide-moi étape par étape, en temps réel, sur ce que je dois cliquer et faire dans mon navigateur.

## Contexte du projet

Je construis un système qui filme le public d'un humoriste, détecte les émotions sur les visages en temps réel, et calcule un score pour l'humoriste.

- Dataset : Human Face Emotions, 9 400 images, 8 classes (anger, content, disgust, fear, happy, neutral, sad, surprise), déjà hébergé sur Roboflow
- Workspace Roboflow : louiss-workspace-jmpoo
- Project ID : human-face-emotions-avush
- Version : 1
- Clé API Roboflow : uro9CATes8iJ8Hd5fLj7
- Modèles à entraîner : YOLOv8n, YOLOv8s, YOLOv8m (3 modèles)
- Framework : Ultralytics YOLO

## Ce que tu dois me guider à faire sur Kaggle

### Étape 1 — Créer un compte Kaggle (si pas déjà fait)
Guide-moi sur kaggle.com pour créer un compte gratuit ou me connecter.

### Étape 2 — Activer le GPU
Dans Kaggle, je dois créer un nouveau notebook et activer l'accélérateur GPU T4. Guide-moi pour trouver le bouton dans l'interface Kaggle.

### Étape 3 — Ajouter ma clé API Roboflow en secret Kaggle
Sur Kaggle, il faut ajouter ma clé API comme "secret" (Settings → Add-ons → Secrets) pour ne pas l'exposer dans le code. Le nom du secret doit être ROBOFLOW_API_KEY et la valeur uro9CATes8iJ8Hd5fLj7.

### Étape 4 — Uploader et exécuter le notebook
J'ai un fichier notebook prêt : kaggle_train_emotions.ipynb. Guide-moi pour l'uploader dans Kaggle et l'exécuter cellule par cellule.

### Étape 5 — Modifier la cellule de la clé API
Dans la cellule 3 du notebook, je dois remplacer "YOUR_API_KEY" par mon secret Kaggle. Montre-moi comment lire le secret depuis l'environnement Kaggle avec ce code :
```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
ROBOFLOW_API_KEY = secrets.get_secret("ROBOFLOW_API_KEY")
```

### Étape 6 — Lancer l'entraînement et attendre
L'entraînement des 3 modèles prendra environ 5-6 heures sur GPU T4. Dis-moi comment vérifier que ça tourne bien, et ce que je dois faire si la session se coupe.

### Étape 7 — Télécharger les résultats
À la fin, je dois télécharger le fichier emotion_scorer_results.zip qui contient :
- Les poids .pt des 3 modèles entraînés
- training_results.json (métriques comparatives)
- model_comparison.png (graphique comparatif)

Guide-moi pour trouver ce fichier dans le panneau Files de Kaggle et le télécharger.

## Ce que j'attends de toi

- Dis-moi exactement où cliquer dans l'interface Kaggle
- Si quelque chose ne marche pas (erreur GPU, session coupée, etc.), propose un fix immédiatement
- À la fin, dis-moi quel modèle est le meilleur et donne-moi la commande exacte pour tester en temps réel sur ma webcam :
  python predict_emotions.py --model NOM_DU_MEILLEUR_MODELE.pt --source 0

Commence par me demander si j'ai déjà un compte Kaggle ou non.
