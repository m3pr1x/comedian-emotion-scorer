"""
Script de téléchargement du dataset Human Face Emotions depuis Roboflow
Dataset: 9400 images, 8 classes (anger, content, disgust, fear, happy, neutral, sad, surprise)
Format: YOLOv8
"""

import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()


def download_dataset():
    """Télécharge le dataset depuis Roboflow en format YOLOv8"""
    api_key = os.environ.get('ROBOFLOW_API_KEY')
    if not api_key:
        raise EnvironmentError(
            "ROBOFLOW_API_KEY manquante. Créez un fichier .env avec:\n"
            "ROBOFLOW_API_KEY=votre_clé_ici"
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("louiss-workspace-jmpoo").project("human-face-emotions-avush")
    version = project.version(1)
    dataset = version.download("yolov8")

    print(f"Dataset téléchargé dans: {dataset.location}")
    return dataset.location


if __name__ == "__main__":
    location = download_dataset()
    print(f"✅ Dataset prêt dans: {location}")
