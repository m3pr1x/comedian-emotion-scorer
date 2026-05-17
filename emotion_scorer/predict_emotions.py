"""
Script de prédiction en temps réel pour la notation d'un humoriste
basé sur les émotions des spectateurs détectées par caméra
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path
from collections import deque, Counter
from ultralytics import YOLO


# Poids des émotions pour le score final
EMOTION_WEIGHTS = {
    "happy": +1.0,       # Très positif
    "content": +0.7,     # Positif
    "surprise": +0.3,    # Légèrement positif (attention captée)
    "neutral": 0.0,      # Neutre
    "sad": -0.5,         # Négatif
    "fear": -0.5,        # Négatif
    "anger": -1.0,       # Très négatif
    "disgust": -1.0,     # Très négatif
}

# Couleurs des bounding boxes par émotion (BGR)
EMOTION_COLORS = {
    "happy": (0, 255, 0),       # Vert
    "content": (0, 200, 100),   # Vert clair
    "surprise": (0, 255, 255),  # Jaune
    "neutral": (200, 200, 200), # Gris
    "sad": (255, 100, 0),       # Bleu-orange
    "fear": (100, 0, 255),      # Violet
    "anger": (0, 0, 255),       # Rouge
    "disgust": (0, 50, 255),    # Rouge foncé
}


class EmotionScorer:
    """Calcule et maintient le score d'un humoriste basé sur les émotions"""
    
    def __init__(self, window_size: int = 150):
        """
        Args:
            window_size: Nombre de frames pour la moyenne glissante
        """
        self.emotion_history = deque(maxlen=window_size)
        self.frame_scores = deque(maxlen=window_size)
        self.total_detections = 0
    
    def update(self, detections: list) -> float:
        """
        Met à jour le score avec les nouvelles détections.
        
        Args:
            detections: Liste d'émotions détectées dans la frame
        
        Returns:
            Score normalisé entre -1 et +1
        """
        if not detections:
            return self.current_score
        
        frame_score = np.mean([EMOTION_WEIGHTS.get(e, 0) for e in detections])
        self.frame_scores.append(frame_score)
        self.emotion_history.extend(detections)
        self.total_detections += len(detections)
        
        return self.current_score
    
    @property
    def current_score(self) -> float:
        """Score courant normalisé [-1, +1]"""
        if not self.frame_scores:
            return 0.0
        return float(np.mean(self.frame_scores))
    
    @property
    def score_percentage(self) -> float:
        """Score en pourcentage [0, 100]"""
        return (self.current_score + 1) / 2 * 100
    
    @property
    def emotion_distribution(self) -> dict:
        """Distribution des émotions détectées"""
        if not self.emotion_history:
            return {}
        counter = Counter(self.emotion_history)
        total = sum(counter.values())
        return {emotion: count/total*100 for emotion, count in counter.most_common()}


def draw_overlay(frame, detections_with_boxes, scorer, fps):
    """
    Dessine les bounding boxes, émotions et le score sur la frame.
    
    Args:
        frame: Image OpenCV
        detections_with_boxes: Liste de (émotion, confiance, bbox)
        scorer: Instance de EmotionScorer
        fps: FPS courant
    
    Returns:
        Frame annotée
    """
    h, w = frame.shape[:2]
    
    # Dessiner les détections
    for emotion, conf, (x1, y1, x2, y2) in detections_with_boxes:
        color = EMOTION_COLORS.get(emotion, (128, 128, 128))
        
        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label avec fond
        label = f"{emotion} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1-lh-8), (x1+lw+4, y1), color, -1)
        cv2.putText(frame, label, (x1+2, y1-4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Panel de score en haut
    panel_height = 120
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    
    score = scorer.current_score
    score_pct = scorer.score_percentage
    
    # Couleur du score
    if score >= 0.5:
        score_color = (0, 255, 0)
    elif score >= 0:
        score_color = (0, 200, 100)
    elif score >= -0.5:
        score_color = (0, 100, 255)
    else:
        score_color = (0, 0, 255)
    
    # Texte du score
    cv2.putText(panel, f"SCORE: {score_pct:.1f}%", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 2)
    cv2.putText(panel, f"Detections totales: {scorer.total_detections} | FPS: {fps:.1f}", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Barre de score
    bar_x = int(score_pct / 100 * (w - 20))
    cv2.rectangle(panel, (10, 85), (w-10, 110), (60, 60, 60), -1)
    cv2.rectangle(panel, (10, 85), (10 + bar_x, 110), score_color, -1)
    cv2.putText(panel, "-1", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    cv2.putText(panel, "+1", (w-25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)
    
    # Distribution des émotions (droite)
    dist = scorer.emotion_distribution
    y_offset = 10
    for emotion, pct in list(dist.items())[:5]:
        color = EMOTION_COLORS.get(emotion, (128, 128, 128))
        text = f"{emotion}: {pct:.1f}%"
        cv2.putText(panel, text, (w-200, y_offset+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 22
    
    # Combiner le panel et la frame
    result = np.vstack([panel, frame])
    return result


def run_inference(
    model_path: str,
    source: str = "0",
    conf_threshold: float = 0.4,
    output_video: str = None
):
    """
    Lance l'inférence en temps réel sur une vidéo ou webcam.
    
    Args:
        model_path: Chemin vers le modèle .pt entraîné
        source: Source vidéo ("0" pour webcam, ou chemin vers une vidéo)
        conf_threshold: Seuil de confiance pour les détections
        output_video: Chemin de sortie pour sauvegarder la vidéo annotée
    """
    print(f"\n🎥 Chargement du modèle: {model_path}")
    model = YOLO(model_path)
    
    print(f"📹 Source: {source}")
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la source: {source}")
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30
    
    writer = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps_input, (w, h + 120))
    
    scorer = EmotionScorer(window_size=150)
    fps_counter = deque(maxlen=30)
    
    print("\n▶️  Démarrage - Appuyez sur 'q' pour quitter, 's' pour sauvegarder le score\n")
    
    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        # Inférence
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Extraire les détections
        detections_with_boxes = []
        emotions_frame = []
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                emotion = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections_with_boxes.append((emotion, conf, (x1, y1, x2, y2)))
                emotions_frame.append(emotion)
        
        # Mettre à jour le score
        scorer.update(emotions_frame)
        
        # FPS
        fps_counter.append(time.time() - t_start)
        fps = 1.0 / (sum(fps_counter) / len(fps_counter))
        
        # Dessiner l'overlay
        annotated = draw_overlay(frame, detections_with_boxes, scorer, fps)
        
        if writer:
            writer.write(annotated)
        
        cv2.imshow("Emotion Scorer - Appuyez Q pour quitter", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            score_data = {
                "final_score": scorer.current_score,
                "score_percentage": scorer.score_percentage,
                "total_detections": scorer.total_detections,
                "emotion_distribution": scorer.emotion_distribution,
            }
            import json
            fname = f"score_{int(time.time())}.json"
            with open(fname, "w") as f:
                json.dump(score_data, f, indent=2)
            print(f"💾 Score sauvegardé: {fname}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Résultat final
    print("\n" + "="*50)
    print(f"🎤 SCORE FINAL DE L'HUMORISTE: {scorer.score_percentage:.1f}%")
    print(f"   Score normalisé: {scorer.current_score:.3f}")
    print(f"   Détections totales: {scorer.total_detections}")
    print("\nDistribution des émotions:")
    for emotion, pct in scorer.emotion_distribution.items():
        print(f"   {emotion:<12} {pct:>6.1f}%")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scorer d'humoriste par détection d'émotions")
    parser.add_argument("--model", type=str, required=True,
                       help="Chemin vers le modèle .pt (ex: runs/emotion_detection/YOLOv8s/weights/best.pt)")
    parser.add_argument("--source", type=str, default="0",
                       help="Source vidéo: '0' pour webcam, ou chemin vers une vidéo")
    parser.add_argument("--conf", type=float, default=0.4,
                       help="Seuil de confiance (défaut: 0.4)")
    parser.add_argument("--output", type=str, default=None,
                       help="Chemin de sortie pour la vidéo annotée (optionnel)")
    
    args = parser.parse_args()
    run_inference(args.model, args.source, args.conf, args.output)
