"""Application Streamlit — Comedian Emotion Scorer.

Lance via : python scripts/main.py  OU  streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── imports internes (PYTHONPATH=src/) ────────────────────────────────────────
from config import (
    EMOTION_CLASSES,
    EMOTION_COLORS,
    EMOTION_DATASET_DIR,
    EMOTION_WEIGHTS,
    MODEL_METRICS_FILE,
    MODELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rgb(name: str) -> str:
    r, g, b = EMOTION_COLORS[name]
    return f"rgb({r},{g},{b})"


def _available_models() -> dict:
    return {k: v for k, v in MODELS.items() if Path(v["path"]).exists()}


def _score_to_pct(score: float) -> float:
    return (score + 1) / 2 * 100


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 1 — Présentation
# ─────────────────────────────────────────────────────────────────────────────

def _tab_presentation() -> None:
    st.header("🎤 Comedian Emotion Scorer")
    st.markdown(
        """
        **Objectif business :** fournir aux humoristes, producteurs et directeurs de salle
        un KPI objectif basé sur les réactions du public.

        Le système filme l'audience, détecte les émotions sur chaque visage en temps réel
        et calcule un **score de 0 % (bide total) à 100 % (standing ovation)**.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Images d'entraînement", "9 400")
    col2.metric("Classes d'émotions", "8")
    col3.metric("Modèles comparés", "3")

    st.divider()
    st.subheader("Système de score")

    rows = []
    for emotion in EMOTION_CLASSES:
        w = EMOTION_WEIGHTS[emotion]
        bar = "█" * int(abs(w) * 10)
        rows.append({
            "Émotion": emotion,
            "Poids":   f"{w:+.1f}",
            "Impact":  ("🟢 Positif" if w > 0 else "🔴 Négatif" if w < 0 else "⚪ Neutre"),
            "Barre":   bar,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Architecture du pipeline")
    st.markdown(
        """
        ```
        📷 Caméra / Vidéo
              │
              ▼
        🔍 Modèle YOLO  →  Détection des visages + classification émotion
              │
              ▼
        📊 EmotionScorer  →  Moyenne pondérée sur fenêtre glissante 150 frames
              │
              ▼
        🎯 Score 0–100%  affiché en temps réel
        ```
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 2 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _tab_dataset() -> None:
    import matplotlib.pyplot as plt

    st.header("📊 Exploration du Dataset")
    st.markdown(
        f"**Source :** Roboflow — Human Face Emotions  \n"
        f"**Split :** 70 % train · 20 % validation · 10 % test  \n"
        f"**Chemin local :** `{EMOTION_DATASET_DIR}`"
    )

    if not EMOTION_DATASET_DIR.exists():
        st.warning(
            "Dataset non trouvé localement. Lance `python emotion_scorer/download_dataset.py` "
            "pour le télécharger."
        )
        return

    # Distribution par split
    try:
        from data import load_class_distribution
        splits = {"train": "Train (70%)", "valid": "Validation (20%)", "test": "Test (10%)"}
        tabs = st.tabs(list(splits.values()))

        for tab, (split, label) in zip(tabs, splits.items()):
            with tab:
                dist = load_class_distribution(split)
                if not dist:
                    st.info("Données indisponibles pour ce split.")
                    continue

                fig, ax = plt.subplots(figsize=(9, 4))
                colors = [_rgb(e) for e in dist.keys()]
                bars = ax.bar(dist.keys(), dist.values(), color=colors, edgecolor="white")
                ax.bar_label(bars, padding=3, fontsize=10)
                ax.set_title(f"Distribution des classes — {label}", fontsize=13)
                ax.set_ylabel("Nombre d'annotations")
                ax.set_ylim(0, max(dist.values()) * 1.15)
                ax.tick_params(axis="x", rotation=15)
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                total = sum(dist.values())
                st.caption(f"Total annotations : {total:,}")
    except Exception as e:
        st.error(f"Erreur lors du chargement des stats : {e}")

    # Exemples d'images
    st.divider()
    st.subheader("Exemples par classe")
    try:
        from data import get_sample_image_paths
        samples = get_sample_image_paths(n_per_class=1)
        cols = st.columns(8)
        for col, emotion in zip(cols, EMOTION_CLASSES):
            paths = samples.get(emotion, [])
            with col:
                st.caption(emotion)
                if paths:
                    st.image(paths[0], use_container_width=True)
                else:
                    st.write("—")
    except Exception as e:
        st.caption(f"Aperçu indisponible : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 3 — Modèles
# ─────────────────────────────────────────────────────────────────────────────

def _tab_models() -> None:
    st.header("🤖 Modèles Comparés")

    model_info = [
        {
            "Modèle":        "YOLOv8n",
            "Architecture":  "CNN (CSPDarknet + FPN)",
            "Tâche":         "Détection + Classification",
            "Paramètres":    "3.2 M",
            "Avantages":     "Ultra-rapide (~2 ms/img GPU), temps réel webcam",
            "Limites":       "Précision inférieure aux modèles plus lourds",
        },
        {
            "Modèle":        "YOLOv8s",
            "Architecture":  "CNN (CSPDarknet + FPN)",
            "Tâche":         "Détection + Classification",
            "Paramètres":    "11.2 M",
            "Avantages":     "Meilleur compromis vitesse / précision",
            "Limites":       "Légèrement plus lent que Nano",
        },
        {
            "Modèle":        "YOLOv8m",
            "Architecture":  "CNN (CSPDarknet + FPN)",
            "Tâche":         "Détection + Classification",
            "Paramètres":    "25.9 M",
            "Avantages":     "Meilleure mAP, robuste aux petits visages",
            "Limites":       "Plus lent — moins adapté temps réel sans GPU dédié",
        },
    ]

    st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Statut des modèles")

    available = _available_models()
    for key, cfg in MODELS.items():
        exists = key in available
        icon   = "✅" if exists else "⏳"
        path   = Path(cfg["path"])
        st.markdown(f"{icon} **{cfg['name']}** — `models/{path.name}`")
        if not exists:
            st.caption(
                f"   → Place le fichier `{path.name}` dans le dossier `models/` "
                "après téléchargement depuis Kaggle."
            )

    if not available:
        st.info(
            "Aucun modèle disponible localement. Les modèles sont en cours d'entraînement "
            "sur Kaggle (notebooks `kaggle_train_emotions.ipynb`). "
            "Télécharge les fichiers `.pt` et place-les dans `models/`."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 4 — Résultats
# ─────────────────────────────────────────────────────────────────────────────

def _tab_results() -> None:
    import matplotlib.pyplot as plt

    st.header("📈 Résultats de l'Évaluation")

    if MODEL_METRICS_FILE.exists():
        df = pd.read_csv(MODEL_METRICS_FILE)
        st.success(f"Résultats chargés depuis `{MODEL_METRICS_FILE.relative_to(MODEL_METRICS_FILE.parent.parent)}`")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Graphique comparatif
        metric_cols = [c for c in df.columns if c not in ("model_key", "model_name", "model_path")]
        if metric_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(df))
            width = 0.8 / len(metric_cols)
            for i, metric in enumerate(metric_cols):
                ax.bar(x + i * width, df[metric], width, label=metric, alpha=0.85)
            ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
            ax.set_xticklabels(df.get("model_name", df["model_key"]))
            ax.set_ylim(0, 1.1)
            ax.legend(loc="upper right")
            ax.set_title("Comparaison des métriques par modèle")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        best_row = df.loc[df["f1_macro"].idxmax()] if "f1_macro" in df.columns else df.iloc[0]
        st.success(f"🏆 Meilleur modèle : **{best_row.get('model_name', best_row['model_key'])}**")
    else:
        import json
        training_json = Path(__file__).parent.parent / "emotion_scorer" / "training_results.json"
        if training_json.exists():
            with open(training_json) as f:
                results = json.load(f)

            st.subheader("Résultats d'entraînement Kaggle (GPU T4)")

            df_train = pd.DataFrame([{
                "Modèle":       r["model_name"],
                "Description":  r["description"],
                "mAP50":        round(r["mAP50"], 4),
                "mAP50-95":     round(r["mAP50_95"], 4),
                "Précision":    round(r["precision"], 4),
                "Rappel":       round(r["recall"], 4),
                "Temps entraîn.": r["training_time_formatted"],
            } for r in results])

            st.dataframe(df_train, use_container_width=True, hide_index=True)

            # Graphique comparatif
            fig, ax = plt.subplots(figsize=(10, 4))
            metrics  = ["mAP50", "mAP50-95", "Précision", "Rappel"]
            x = np.arange(len(df_train))
            width = 0.8 / len(metrics)
            for i, metric in enumerate(metrics):
                ax.bar(x + i * width, df_train[metric], width, label=metric, alpha=0.85)
            ax.set_xticks(x + width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(df_train["Modèle"])
            ax.set_ylim(0, 1.0)
            ax.legend(loc="lower right")
            ax.set_title("Comparaison des métriques — entraînement Kaggle T4")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            best = max(results, key=lambda r: r["mAP50"])
            st.success(f"🏆 Meilleur modèle : **{best['model_name']}** — mAP50 = {best['mAP50']:.4f}")
        else:
            st.info(
                "Lance `python scripts/main.py` une fois les modèles `.pt` placés dans `models/` "
                "pour générer `results/model_metrics.csv`."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 5 — Démo
# ─────────────────────────────────────────────────────────────────────────────

def _infer_crowd(model, img_path: str, conf: float) -> list[dict]:
    """Pipeline 2 étapes : détecteur de visages OpenCV → crop → YOLO émotion sur chaque visage.
    Retourne une liste de dicts avec emotion, conf et bbox (x1,y1,x2,y2) pour l'annotation."""
    import cv2, tempfile, os

    img     = cv2.imread(img_path)
    h, w    = img.shape[:2]
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray    = cv2.equalizeHist(gray)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    min_face = max(40, min(h, w) // 12)
    faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(min_face, min_face))

    if len(faces) == 0:
        return []

    detections = []
    model_names = None

    for (fx, fy, fw, fh) in faces:
        pad_x = int(fw * 0.20)
        pad_y = int(fh * 0.20)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y)

        crop = img[y1:y2, x1:x2]

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
            cv2.imwrite(t.name, crop)
            crop_path = t.name

        result = model.model(crop_path, conf=conf, verbose=False)[0]
        os.unlink(crop_path)

        if model_names is None:
            model_names = result.names

        if result.boxes and len(result.boxes) > 0:
            best   = max(result.boxes, key=lambda b: float(b.conf[0]))
            cls_id = int(best.cls[0])
            detections.append({
                "emotion": model_names.get(cls_id, "unknown"),
                "conf":    float(best.conf[0]),
                "bbox":    (fx, fy, fx + fw, fy + fh),
            })

    return detections


def _draw_detections(img_path: str, detections: list[dict]) -> str:
    """Dessine les bounding boxes et labels sur l'image, retourne le chemin vers l'image annotée."""
    import cv2, tempfile

    img = cv2.imread(img_path)
    for det in detections:
        emotion  = det["emotion"]
        conf_val = det["conf"]
        x1, y1, x2, y2 = det["bbox"]
        r, g, b  = EMOTION_COLORS.get(emotion, (128, 128, 128))
        color_bgr = (b, g, r)

        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{emotion} {conf_val:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
        cv2.imwrite(t.name, img)
        return t.name


def _tab_demo() -> None:
    st.header("🎯 Démo — Analyse d'Émotion")

    available = _available_models()

    if not available:
        st.warning(
            "Aucun modèle disponible. Place les fichiers `.pt` dans `models/` "
            "pour activer la démo."
        )
        st.markdown("**Commande pour tester en temps réel avec la webcam :**")
        st.code(
            "python emotion_scorer/predict_emotions.py \\\n"
            "  --model models/yolov8s_emotions.pt \\\n"
            "  --source 0",
            language="bash"
        )
        return

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        model_key = st.selectbox(
            "Modèle",
            options=list(available.keys()),
            format_func=lambda k: available[k]["name"],
            index=min(2, len(available) - 1),
        )
    with col_s2:
        conf = st.slider("Seuil de confiance", 0.1, 0.9, 0.25, 0.05,
                         help="Baisse ce seuil si aucun visage n'est détecté")
    with col_s3:
        iou = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05,
                        help="Baisse pour supprimer les détections en double")
    with col_s4:
        mode_foule = st.toggle("Mode Foule 👥", value=False,
                               help="Détecte chaque visage individuellement via OpenCV puis classifie l'émotion — idéal pour les groupes")

    uploaded = st.file_uploader("Charge une image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(tmp_path, caption="Image uploadée", use_container_width=True)

        with col_res:
            with st.spinner("Inférence en cours..."):
                try:
                    from model_io import YOLOEmotionClassifier
                    model = YOLOEmotionClassifier(Path(available[model_key]["path"]), conf=conf)

                    if mode_foule:
                        detections = _infer_crowd(model, tmp_path, conf)
                    else:
                        result = model.model(tmp_path, conf=conf, iou=iou, verbose=False)[0]
                        model_names = result.names
                        detections = []
                        if result.boxes:
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                detections.append({
                                    "emotion": model_names.get(cls_id, "unknown"),
                                    "conf": float(box.conf[0]),
                                })

                    if not detections:
                        st.warning("Aucun visage détecté. Essaie de baisser le seuil ou active le Mode Foule.")
                    else:
                        st.success(f"{len(detections)} visage(s) détecté(s)")
                        detected_emotions = []
                        for i, det in enumerate(detections):
                            emotion  = det["emotion"]
                            conf_val = det["conf"]
                            weight   = EMOTION_WEIGHTS.get(emotion, 0.0)
                            score    = _score_to_pct(weight)
                            color    = EMOTION_COLORS.get(emotion, (128, 128, 128))
                            r, g, b  = color
                            detected_emotions.append(emotion)
                            st.markdown(
                                f"<div style='border-left:4px solid rgb({r},{g},{b});padding:8px;margin:4px 0'>"
                                f"<b>Visage {i+1}</b> — {emotion.upper()} "
                                f"({conf_val:.0%} confiance)<br>"
                                f"Contribution au score : <b>{score:.0f}%</b></div>",
                                unsafe_allow_html=True,
                            )

                        scores   = [_score_to_pct(EMOTION_WEIGHTS.get(e, 0.0)) for e in detected_emotions]
                        global_s = float(np.mean(scores))
                        st.metric("Score global de cette frame", f"{global_s:.1f}%")
                except Exception as e:
                    st.error(f"Erreur lors de l'inférence : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 6 — Score Live (simulation)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_score() -> None:
    import matplotlib.pyplot as plt

    st.header("🎤 Simulateur de Score Humoriste")
    st.markdown(
        "Simule le score en temps réel en ajustant la distribution des émotions du public."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Distribution du public")
        sliders = {}
        for emotion in EMOTION_CLASSES:
            val = st.slider(
                emotion,
                0, 100,
                {"happy": 30, "content": 20, "neutral": 25, "surprise": 10}.get(emotion, 5),
                key=f"slider_{emotion}",
            )
            sliders[emotion] = val

        total = sum(sliders.values())
        if total == 0:
            st.warning("Au moins une émotion doit être > 0")
            return

        # Normaliser
        fractions = {e: v / total for e, v in sliders.items()}

    with col2:
        # Calcul du score
        raw_score = sum(fractions[e] * EMOTION_WEIGHTS[e] for e in EMOTION_CLASSES)
        pct_score = _score_to_pct(raw_score)

        # Couleur du score
        if pct_score >= 70:
            color, label = "#2ecc71", "Excellent !"
        elif pct_score >= 50:
            color, label = "#f39c12", "Correct"
        elif pct_score >= 30:
            color, label = "#e67e22", "Décevant"
        else:
            color, label = "#e74c3c", "Catastrophique"

        st.markdown(
            f"<div style='text-align:center;padding:20px;background:{color}20;"
            f"border:2px solid {color};border-radius:12px;margin-bottom:16px'>"
            f"<h1 style='color:{color};margin:0'>{pct_score:.1f}%</h1>"
            f"<p style='color:{color};font-size:1.2em;margin:4px 0'>{label}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Graphique camembert
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        non_zero = {e: v for e, v in fractions.items() if v > 0}
        colors_pie = [tuple(c / 255 for c in EMOTION_COLORS[e]) for e in non_zero]
        axes[0].pie(
            non_zero.values(),
            labels=non_zero.keys(),
            colors=colors_pie,
            autopct="%1.0f%%",
            startangle=90,
        )
        axes[0].set_title("Répartition des émotions")

        # Barre de score avec gradient
        axes[1].barh(["Score"], [pct_score], color=color, height=0.4)
        axes[1].barh(["Score"], [100 - pct_score], left=[pct_score], color="#ecf0f1", height=0.4)
        axes[1].set_xlim(0, 100)
        axes[1].axvline(50, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Score (%)")
        axes[1].set_title("Score humoriste")
        axes[1].text(pct_score / 2, 0, f"{pct_score:.1f}%", ha="center", va="center",
                     fontweight="bold", color="white" if pct_score > 15 else "black")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 6 — Caméra Live
# ─────────────────────────────────────────────────────────────────────────────

def _tab_camera() -> None:
    import cv2
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode

    st.header("📷 Caméra Live — Détection en Temps Réel")

    available = _available_models()
    if not available:
        st.warning("Aucun modèle disponible. Place les fichiers `.pt` dans `models/`.")
        return

    col1, col2 = st.columns(2)
    with col1:
        model_key = st.selectbox(
            "Modèle",
            options=list(available.keys()),
            format_func=lambda k: available[k]["name"],
            index=min(2, len(available) - 1),
            key="cam_model",
        )
    with col2:
        cam_conf = st.slider("Seuil de confiance", 0.1, 0.9, 0.25, 0.05, key="cam_conf")

    model_path = str(available[model_key]["path"])

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Couleurs BGR pour OpenCV
    COLORS_BGR = {e: (b, g, r) for e, (r, g, b) in EMOTION_COLORS.items()}

    class EmotionVideoProcessor:
        def __init__(self):
            from ultralytics import YOLO
            self.model = YOLO(model_path)

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img  = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            min_face = max(40, min(h, w) // 12)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=7, minSize=(min_face, min_face)
            )

            import tempfile, os
            overlay = []

            for (fx, fy, fw, fh) in faces:
                pad_x = int(fw * 0.20)
                pad_y = int(fh * 0.20)
                x1 = max(0, fx - pad_x)
                y1 = max(0, fy - pad_y)
                x2 = min(w, fx + fw + pad_x)
                y2 = min(h, fy + fh + pad_y)
                crop = img[y1:y2, x1:x2]

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
                    cv2.imwrite(t.name, crop)
                    crop_path = t.name

                result = self.model(crop_path, conf=cam_conf, verbose=False)[0]
                os.unlink(crop_path)

                emotion, conf_val = "unknown", 0.0
                if result.boxes and len(result.boxes) > 0:
                    best   = max(result.boxes, key=lambda b: float(b.conf[0]))
                    cls_id = int(best.cls[0])
                    emotion   = result.names.get(cls_id, "unknown")
                    conf_val  = float(best.conf[0])

                overlay.append((fx, fy, fx + fw, fy + fh, emotion, conf_val))

            # Dessine les rectangles et labels sur l'image
            for (bx1, by1, bx2, by2, emotion, conf_val) in overlay:
                color = COLORS_BGR.get(emotion, (128, 128, 128))
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
                label = f"{emotion} {conf_val:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img, (bx1, by1 - th - 6), (bx1 + tw + 4, by1), color, -1)
                cv2.putText(img, label, (bx1 + 2, by1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Score global de la frame
            if overlay:
                emotions = [e for *_, e, _ in overlay]
                raw = sum(EMOTION_WEIGHTS.get(e, 0) for e in emotions) / len(emotions)
                pct = _score_to_pct(raw)
                score_label = f"Score: {pct:.0f}%"
                cv2.putText(img, score_label, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
                cv2.putText(img, score_label, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 100), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # ── Mode Flux Temps Réel ──────────────────────────────────────────────────
    st.subheader("🎥 Flux temps réel")
    st.info("Clique sur **START** pour activer la caméra. Les émotions s'affichent en temps réel avec le score de la frame.")

    webrtc_streamer(
        key="emotion-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ── Mode Snapshot ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📸 Snapshot — Prendre une photo")

    snapshot = st.camera_input("Prends une photo avec ta webcam")

    if snapshot:
        import tempfile
        from model_io import YOLOEmotionClassifier

        img_bytes = snapshot.getvalue()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        with st.spinner("Analyse en cours..."):
            model = YOLOEmotionClassifier(Path(available[model_key]["path"]), conf=cam_conf)
            detections = _infer_crowd(model, tmp_path, cam_conf)

        if not detections:
            st.warning("Aucun visage détecté. Essaie avec une meilleure luminosité ou baisse le seuil.")
        else:
            col_snap, col_info = st.columns([1, 1])
            annotated_path = _draw_detections(tmp_path, detections)

            with col_snap:
                st.image(annotated_path, caption=f"{len(detections)} visage(s) détecté(s)", use_container_width=True)

            with col_info:
                detected_emotions = []
                for i, det in enumerate(detections):
                    emotion  = det["emotion"]
                    conf_val = det["conf"]
                    weight   = EMOTION_WEIGHTS.get(emotion, 0.0)
                    score    = _score_to_pct(weight)
                    color    = EMOTION_COLORS.get(emotion, (128, 128, 128))
                    r, g, b  = color
                    detected_emotions.append(emotion)
                    st.markdown(
                        f"<div style='border-left:4px solid rgb({r},{g},{b});padding:8px;margin:4px 0'>"
                        f"<b>Visage {i+1}</b> — {emotion.upper()} ({conf_val:.0%})<br>"
                        f"Score : <b>{score:.0f}%</b></div>",
                        unsafe_allow_html=True,
                    )

                scores   = [_score_to_pct(EMOTION_WEIGHTS.get(e, 0.0)) for e in detected_emotions]
                global_s = float(np.mean(scores))
                st.metric("Score global de cette frame", f"{global_s:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> None:
    st.set_page_config(
        page_title="Comedian Emotion Scorer",
        page_icon="🎤",
        layout="wide",
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🏠 Présentation",
        "📊 Dataset",
        "🤖 Modèles",
        "📈 Résultats",
        "🎯 Démo",
        "📷 Caméra Live",
        "🎤 Score Live",
    ])

    with tab1: _tab_presentation()
    with tab2: _tab_dataset()
    with tab3: _tab_models()
    with tab4: _tab_results()
    with tab5: _tab_demo()
    with tab6: _tab_camera()
    with tab7: _tab_score()


if __name__ == "__main__":
    build_app()
