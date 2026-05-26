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

_PROJECT_ROOT = Path(__file__).parent.parent
_ASSETS_DIR   = _PROJECT_ROOT / "emotion_scorer" / "assets"


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

    st.markdown("""
    ### Le problème
    Un humoriste monte sur scène. À la fin du show, il a l'impression que ça s'est bien passé…
    mais est-ce vraiment le cas ? Les retours sont **subjectifs** et impossibles à mesurer objectivement.

    **Notre solution :** une caméra filme l'audience, un modèle de vision par ordinateur détecte
    les émotions sur chaque visage en temps réel, et calcule un **score objectif de 0 % à 100 %**.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎭 **Comedy Club**\n\nScore en direct pendant le show — feedback instantané pour l'artiste après chaque blague.")
    with col2:
        st.info("📺 **Talent Show**\n\nKPI objectif pour les jurés — remplace les notes subjectives par une mesure de réaction du public.")
    with col3:
        st.info("🎓 **Coaching**\n\nAnalyse post-show segment par segment pour identifier les passages forts et les creux.")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images d'entraînement", "9 400")
    col2.metric("Classes d'émotions", "8")
    col3.metric("Modèles comparés", "3")
    col4.metric("mAP50 meilleur modèle", "78.3 %")

    # Score scale
    st.divider()
    st.subheader("Échelle de score")
    score_zones = [
        ("#e74c3c", "0 – 25 %",   "Bide total 💀",         "Anger / Disgust dominent"),
        ("#e67e22", "25 – 50 %",  "Décevant 😕",           "Sad / Fear visibles"),
        ("#f39c12", "50 – 70 %",  "Correct 😐",            "Public neutre, attentif"),
        ("#2ecc71", "70 – 90 %",  "Bon show 😄",           "Happy / Content dominants"),
        ("#27ae60", "90 – 100 %", "Standing ovation 🏆",   "Euphorie généralisée"),
    ]
    cols = st.columns(5)
    for col, (color, zone, label, desc) in zip(cols, score_zones):
        with col:
            st.markdown(
                f"<div style='background:{color}20;border:2px solid {color};border-radius:8px;"
                f"padding:10px;text-align:center'>"
                f"<b style='color:{color}'>{zone}</b><br>{label}<br>"
                f"<small style='color:gray'>{desc}</small></div>",
                unsafe_allow_html=True,
            )

    # Pipeline
    st.divider()
    st.subheader("Architecture du pipeline")

    col_pipe, col_detail = st.columns([1, 1])
    with col_pipe:
        st.markdown("""
        ```
        📷  Caméra / Image
               │
               ▼
        🔍  Détecteur de visages
               │   OpenCV Haar Cascade
               │   → localise chaque visage
               ▼
        ✂️   Crop + padding 20 %
               │
               ▼
        🤖  YOLOv8 Emotion Classifier
               │   → prédit émotion + confiance
               │     sur chaque crop
               ▼
        📊  EmotionScorer
               │   → moyenne pondérée
               ▼
        🎯  Score 0 – 100 %
        ```
        """)
    with col_detail:
        st.markdown("""
        **Étape 1 — Localisation des visages**

        OpenCV Haar Cascade scanne la frame et retourne les bounding boxes
        de tous les visages détectés. Chaque région est élargie de 20 %
        pour capturer l'expression complète (sourcils, menton).

        **Étape 2 — Classification de l'émotion**

        Chaque crop est passé dans le modèle YOLOv8 fine-tuné sur 9 400 images
        annotées. Le modèle retourne la classe émotion et un score de confiance.

        **Étape 3 — Score global**

        Chaque émotion a un poids entre −1 (anger / disgust) et +1 (happy).
        La moyenne est normalisée vers [0 %, 100 %] via la formule :

        `score = (raw + 1) / 2 × 100`
        """)

    # Scoring table
    st.divider()
    st.subheader("Système de pondération des émotions")

    rows = []
    for emotion in EMOTION_CLASSES:
        w = EMOTION_WEIGHTS[emotion]
        bar = "█" * int(abs(w) * 10)
        rows.append({
            "Émotion": emotion,
            "Poids":   f"{w:+.1f}",
            "Impact":  ("🟢 Positif" if w > 0 else "🔴 Négatif" if w < 0 else "⚪ Neutre"),
            "Visualisation": bar,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Formule : score_pct = ( (Σ poids_émotion_i / N) + 1 ) / 2 × 100  —  "
        "Avec N = nombre de visages détectés dans la frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 2 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _tab_dataset() -> None:
    import matplotlib.pyplot as plt

    st.header("📊 Exploration du Dataset")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Source :** Human Face Emotions — Roboflow Universe
        **Format d'annotation :** YOLOv8 (fichiers `.txt` avec coordonnées bounding box normalisées)
        **Split :** 70 % train · 20 % validation · 10 % test
        **Augmentations appliquées :** flip horizontal, HSV jitter, mosaic, random erasing
        """)
    with col2:
        st.metric("Images totales", "9 400")
        st.metric("Classes", "8")

    # Class descriptions — toujours visible
    st.divider()
    st.subheader("Les 8 classes d'émotions")

    class_info = [
        {"Émotion": "😠 anger",    "Description FACS": "Sourcils froncés, lèvres serrées, regard intense",          "Poids score": "−1.0",  "Interprétation": "Public irrité — catastrophique"},
        {"Émotion": "😌 content",  "Description FACS": "Sourire doux, joues légèrement relevées, yeux détendus",     "Poids score": "+0.7",  "Interprétation": "Public satisfait — très bon"},
        {"Émotion": "🤢 disgust",  "Description FACS": "Lèvre supérieure retroussée, nez plissé, regard de côté",   "Poids score": "−1.0",  "Interprétation": "Public rebuté — catastrophique"},
        {"Émotion": "😨 fear",     "Description FACS": "Yeux écarquillés, bouche entrouverte, sourcils relevés",     "Poids score": "−0.5",  "Interprétation": "Inconfort dans le public — négatif"},
        {"Émotion": "😄 happy",    "Description FACS": "Sourire large (dents visibles), joues remontées, yeux plissés","Poids score": "+1.0","Interprétation": "Rires — excellent"},
        {"Émotion": "😐 neutral",  "Description FACS": "Expression de repos, muscles faciaux non activés",           "Poids score": "0.0",   "Interprétation": "Écoute attentive — neutre"},
        {"Émotion": "😢 sad",      "Description FACS": "Coins de bouche abaissés, yeux tombants, regard bas",        "Poids score": "−0.5",  "Interprétation": "Public ennuyé — négatif"},
        {"Émotion": "😲 surprise", "Description FACS": "Yeux grands ouverts, sourcils levés, bouche en O",           "Poids score": "+0.3",  "Interprétation": "Réaction vive — légèrement positif"},
    ]
    st.dataframe(pd.DataFrame(class_info), use_container_width=True, hide_index=True)
    st.caption("FACS = Facial Action Coding System — référence scientifique pour décrire les expressions faciales.")

    # Labels distribution image (générée lors de l'entraînement)
    labels_img = (
        _PROJECT_ROOT / "emotion_scorer" / "runs" / "detect" /
        "runs" / "emotion_detection" / "YOLOv8n" / "labels.jpg"
    )
    if labels_img.exists():
        st.divider()
        st.subheader("Distribution des annotations YOLO (train run local)")
        st.image(str(labels_img), caption="Position et taille des bounding boxes dans le dataset", use_container_width=True)

    # Confusion matrix par modèle (si dispo dans assets/)
    _show_assets_section("confusion_matrix", "Matrices de confusion par modèle",
                         "Dépose les fichiers `confusion_matrix_yolov8n.png`, `...s.png`, `...m.png` "
                         "dans `emotion_scorer/assets/` depuis les outputs Kaggle.")

    if not EMOTION_DATASET_DIR.exists():
        st.divider()
        st.warning(
            "Dataset non téléchargé localement — distribution interactive indisponible.  \n"
            "Lance `python emotion_scorer/download_dataset.py` pour le télécharger."
        )
        return

    # Distribution par split
    st.divider()
    st.subheader("Distribution des classes par split")
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


def _show_assets_section(prefix: str, title: str, hint: str) -> None:
    """Affiche les images d'assets si présentes dans emotion_scorer/assets/."""
    if not _ASSETS_DIR.exists():
        return
    imgs = sorted(_ASSETS_DIR.glob(f"{prefix}_*.png")) + sorted(_ASSETS_DIR.glob(f"{prefix}_*.jpg"))
    if not imgs:
        return
    st.divider()
    st.subheader(title)
    cols = st.columns(len(imgs))
    for col, img in zip(cols, imgs):
        with col:
            st.image(str(img), caption=img.stem.replace("_", " "), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 3 — Modèles
# ─────────────────────────────────────────────────────────────────────────────

def _tab_models() -> None:
    st.header("🤖 Modèles — YOLOv8")

    # Pourquoi YOLO ?
    st.subheader("Pourquoi YOLOv8 ?")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        La tâche est une **détection + classification** : il faut localiser les visages **et**
        prédire l'émotion en une seule passe. YOLOv8 est la solution naturelle.

        Les alternatives considérées :
        """)
        alternatives = [
            {"Approche": "YOLOv8 (choisi ✅)",    "Avantages": "Un seul modèle, temps réel, fine-tunable facilement",         "Inconvénients": "Moins précis que ViT sur des expressions subtiles"},
            {"Approche": "MTCNN + ResNet",         "Avantages": "MTCNN très robuste pour la détection de visages",             "Inconvénients": "Pipeline en 2 modèles séparés, latence cumulée"},
            {"Approche": "Vision Transformer (ViT)","Avantages": "Meilleure capture des patterns globaux (expressions fines)", "Inconvénients": "Beaucoup plus lourd, pas temps réel sans GPU dédié"},
            {"Approche": "FER+ (ResNet fine-tuné)", "Avantages": "Spécialisé émotions, bonnes performances",                  "Inconvénients": "Pas de détection intégrée, nécessite un détecteur externe"},
        ]
        st.dataframe(pd.DataFrame(alternatives), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("""
        **YOLO = You Only Look Once**

        Contrairement aux approches en 2 étapes (proposer des régions → classifier),
        YOLO prédit bounding boxes et classes **en une seule passe** sur le réseau.

        → Idéal pour le **temps réel** (webcam 30 fps).
        """)
        st.info("⚡ YOLOv8n : ~2 ms / image sur GPU\n\n🎯 YOLOv8m : mAP50 = 78.3 %")

    # Architecture
    st.divider()
    st.subheader("Architecture YOLOv8")

    with st.expander("📐 Détails de l'architecture (cliquer pour développer)"):
        col_arch1, col_arch2 = st.columns(2)
        with col_arch1:
            st.markdown("""
            **Backbone — CSPDarknet**
            - Réseau convolutif profond avec connexions résiduelles
            - Extrait les features visuelles à plusieurs échelles
            - Pré-entraîné sur COCO (80 classes, 118k images)

            **Neck — FPN (Feature Pyramid Network)**
            - Fusionne les features de différentes résolutions
            - Permet la détection multi-échelle (petits et grands visages)

            **Head — Decoupled Detection Head**
            - Prédit simultanément : position (x, y, w, h), objectness, classe
            - DFL (Distribution Focal Loss) pour la localisation précise
            """)
        with col_arch2:
            st.markdown("""
            **Transfer learning**
            - Point de départ : poids COCO pré-entraînés
            - Fine-tuning complet sur le dataset émotions
            - Batch size 16, image size 640×640

            **Comparaison des variantes**
            """)
            variants = pd.DataFrame([
                {"Variante": "YOLOv8n (Nano)",   "Paramètres": "3.2 M",  "Taille .pt": "~6 MB",  "Vitesse GPU": "~2 ms"},
                {"Variante": "YOLOv8s (Small)",  "Paramètres": "11.2 M", "Taille .pt": "~21 MB", "Vitesse GPU": "~4 ms"},
                {"Variante": "YOLOv8m (Medium)", "Paramètres": "25.9 M", "Taille .pt": "~50 MB", "Vitesse GPU": "~8 ms"},
            ])
            st.dataframe(variants, use_container_width=True, hide_index=True)

    # Training setup
    st.divider()
    st.subheader("Configuration d'entraînement")

    col_train1, col_train2 = st.columns(2)
    with col_train1:
        st.markdown("**Hyperparamètres**")
        train_config = pd.DataFrame([
            {"Paramètre": "Epochs",           "Valeur": "50"},
            {"Paramètre": "Batch size",        "Valeur": "16"},
            {"Paramètre": "Image size",        "Valeur": "640 × 640"},
            {"Paramètre": "Optimizer",         "Valeur": "Auto (Adam → SGD)"},
            {"Paramètre": "Learning rate",     "Valeur": "0.01 → 0.01 (cosine)"},
            {"Paramètre": "Patience (early stopping)", "Valeur": "15 epochs"},
            {"Paramètre": "Pretrained",        "Valeur": "Oui (COCO)"},
        ])
        st.dataframe(train_config, use_container_width=True, hide_index=True)

    with col_train2:
        st.markdown("**Augmentations de données**")
        augments = pd.DataFrame([
            {"Augmentation": "Flip horizontal",    "Valeur": "50 % des images"},
            {"Augmentation": "HSV hue jitter",     "Valeur": "±1.5 %"},
            {"Augmentation": "HSV saturation",     "Valeur": "±70 %"},
            {"Augmentation": "HSV value (brightness)", "Valeur": "±40 %"},
            {"Augmentation": "Scale",              "Valeur": "±50 %"},
            {"Augmentation": "Mosaic",             "Valeur": "100 % (combine 4 images)"},
            {"Augmentation": "Random erasing",     "Valeur": "40 %"},
        ])
        st.dataframe(augments, use_container_width=True, hide_index=True)

    st.info("🖥️ **Infrastructure :** Kaggle Notebook — GPU NVIDIA Tesla T4 (16 GB VRAM)")

    # Training batch examples
    batch_dir = (
        _PROJECT_ROOT / "emotion_scorer" / "runs" / "detect" /
        "runs" / "emotion_detection" / "YOLOv8n"
    )
    batch_imgs = [batch_dir / f"train_batch{i}.jpg" for i in range(3) if (batch_dir / f"train_batch{i}.jpg").exists()]
    if batch_imgs:
        st.divider()
        st.subheader("Exemples de batches d'entraînement (YOLOv8n)")
        st.caption("Chaque image montre un batch de 16 images avec leurs annotations de bounding boxes.")
        cols = st.columns(len(batch_imgs))
        for col, img in zip(cols, batch_imgs):
            with col:
                st.image(str(img), use_container_width=True)

    # Model comparison image
    comp_img = _PROJECT_ROOT / "emotion_scorer" / "model_comparison.png"
    if comp_img.exists():
        st.divider()
        st.subheader("Comparaison visuelle des modèles")
        st.image(str(comp_img), caption="Comparaison générée par compare_models.py", use_container_width=True)

    # Statut des modèles
    st.divider()
    st.subheader("Statut des modèles locaux")

    available = _available_models()
    for key, cfg in MODELS.items():
        exists = key in available
        icon   = "✅" if exists else "⏳"
        path   = Path(cfg["path"])
        size_str = ""
        if exists:
            size_mb = path.stat().st_size / 1_000_000
            size_str = f" ({size_mb:.0f} MB)"
        st.markdown(f"{icon} **{cfg['name']}** — `models/{path.name}`{size_str}")
        if not exists:
            st.caption(f"   → Place le fichier `{path.name}` dans `models/` depuis Kaggle.")

    if not available:
        st.info(
            "Aucun modèle disponible localement. "
            "Télécharge les fichiers `.pt` depuis le notebook Kaggle et place-les dans `models/`."
        )


def _show_training_artifacts() -> None:
    """Affiche les courbes d'entraînement, matrices de confusion et courbes PR par modèle."""
    if not _ASSETS_DIR.exists():
        return

    model_variants = [
        ("yolov8n", "YOLOv8n — Nano"),
        ("yolov8s", "YOLOv8s — Small"),
        ("yolov8m", "YOLOv8m — Medium"),
    ]

    # Garde seulement les modèles qui ont au moins un fichier disponible
    available_variants = []
    for key, label in model_variants:
        has_any = any([
            (_ASSETS_DIR / f"results_{key}.png").exists(),
            (_ASSETS_DIR / f"confusion_matrix_{key}.png").exists(),
            (_ASSETS_DIR / f"pr_curve_{key}.png").exists(),
        ])
        if has_any:
            available_variants.append((key, label))

    if not available_variants:
        return

    st.divider()
    st.subheader("📊 Visualisations d'entraînement par modèle")
    st.caption(
        "Pour ajouter les visualisations des autres modèles, dépose leurs images dans "
        "`emotion_scorer/assets/` avec le suffixe `_yolov8s` ou `_yolov8m`."
    )

    tab_labels = [label for _, label in available_variants]
    tabs = st.tabs(tab_labels)

    artifact_meta = {
        "results": {
            "title": "📈 Courbes d'entraînement",
            "caption": (
                "On voit ici l'évolution du modèle sur 50 epochs. "
                "Les deux courbes de loss descendent régulièrement — le modèle apprend. "
                "Le mAP50 monte et se stabilise autour de **78 %**, ce qui confirme une bonne convergence "
                "sans surapprentissage."
            ),
        },
        "confusion_matrix": {
            "title": "🔲 Matrice de confusion",
            "caption": (
                "La diagonale représente les bonnes prédictions. "
                "**Happy** et **anger** sont très bien reconnus — ce sont les émotions avec les signaux "
                "faciaux les plus marqués. "
                "La principale confusion se situe entre **content** et **neutral**, "
                "deux expressions proches visuellement."
            ),
        },
        "pr_curve": {
            "title": "📉 Courbe Précision / Rappel",
            "caption": (
                "Plus l'aire sous la courbe est grande, meilleur est le modèle. "
                "On choisit un seuil de confiance à **0.25** dans la démo — "
                "c'est le point qui équilibre le mieux la détection sans générer trop de faux positifs."
            ),
        },
    }

    for tab, (key, label) in zip(tabs, available_variants):
        with tab:
            for prefix, meta in artifact_meta.items():
                img_path = _ASSETS_DIR / f"{prefix}_{key}.png"
                if not img_path.exists():
                    continue
                st.markdown(f"**{meta['title']}**")
                st.image(str(img_path), use_container_width=True)
                st.info(meta["caption"])
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 4 — Résultats
# ─────────────────────────────────────────────────────────────────────────────

def _tab_results() -> None:
    import matplotlib.pyplot as plt

    st.header("📈 Résultats de l'Évaluation")

    # Glossaire des métriques
    with st.expander("📖 Glossaire des métriques (cliquer pour développer)"):
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown("""
            **mAP50** *(Mean Average Precision @ IoU 0.5)*
            Précision moyenne sur toutes les classes avec un seuil d'IoU de 0.5.
            C'est **la métrique principale** en détection d'objets.

            **mAP50-95**
            mAP calculée pour des seuils IoU de 0.5 à 0.95 (plus stricte).
            Mesure la qualité de la localisation des bounding boxes.

            **Précision**
            Parmi les prédictions positives, combien sont correctes ?
            → Faible précision = beaucoup de fausses détections.
            """)
        with col_g2:
            st.markdown("""
            **Rappel** *(Recall)*
            Parmi les vrais positifs, combien ont été détectés ?
            → Faible rappel = visages ou émotions manqués.

            **F1-Macro**
            Moyenne harmonique précision/rappel, équilibrée sur toutes les classes.
            Utile quand les classes sont déséquilibrées.

            **Accuracy**
            Taux de bonne classification sur le jeu de test
            (calculé sur les images du dataset Roboflow).
            """)

    if MODEL_METRICS_FILE.exists():
        df = pd.read_csv(MODEL_METRICS_FILE)
        st.success(f"Résultats chargés depuis `results/model_metrics.csv` (évaluation locale)")

        # Tableau formaté
        df_display = df.copy()
        metric_cols = [c for c in df.columns if c not in ("model_key", "model_name", "model_path")]
        for col in metric_cols:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.3f}")
        st.dataframe(df_display.drop(columns=["model_path"], errors="ignore"),
                     use_container_width=True, hide_index=True)

        # Radar chart interactif
        st.divider()
        st.subheader("Radar chart — comparaison multi-métriques")
        try:
            import plotly.graph_objects as go
            radar_metrics = [c for c in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
                             if c in df.columns]
            fig = go.Figure()
            colors_radar = ["#3498db", "#e74c3c", "#2ecc71"]
            for i, (_, row) in enumerate(df.iterrows()):
                values = [float(row[m]) for m in radar_metrics]
                values += [values[0]]  # ferme le polygone
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=radar_metrics + [radar_metrics[0]],
                    fill="toself",
                    name=row.get("model_name", row["model_key"]),
                    line_color=colors_radar[i % len(colors_radar)],
                    opacity=0.7,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0.6, 0.75])),
                showlegend=True,
                title="Comparaison radar des métriques d'évaluation",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.caption("Plotly non disponible — graphique radar désactivé.")

        # Graphique barres
        st.divider()
        st.subheader("Comparaison par métrique")
        if metric_cols:
            fig, ax = plt.subplots(figsize=(10, 4))
            x = np.arange(len(df))
            width = 0.8 / len(metric_cols)
            for i, metric in enumerate(metric_cols):
                ax.bar(x + i * width, df[metric].astype(float), width, label=metric, alpha=0.85)
            ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
            ax.set_xticklabels(df.get("model_name", df["model_key"]))
            ax.set_ylim(0, 1.1)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title("Comparaison des métriques par modèle")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        best_row = df.loc[df["f1_macro"].astype(float).idxmax()] if "f1_macro" in df.columns else df.iloc[0]
        st.success(f"🏆 Meilleur modèle : **{best_row.get('model_name', best_row['model_key'])}**")

    else:
        import json
        training_json = _PROJECT_ROOT / "emotion_scorer" / "training_results.json"
        if training_json.exists():
            with open(training_json) as f:
                results = json.load(f)

            st.subheader("Résultats d'entraînement Kaggle (GPU Tesla T4)")
            st.caption("Ces métriques proviennent de la validation YOLO pendant l'entraînement sur Kaggle.")

            df_train = pd.DataFrame([{
                "Modèle":           r["model_name"],
                "Description":      r["description"],
                "mAP50":            round(r["mAP50"], 4),
                "mAP50-95":         round(r["mAP50_95"], 4),
                "Précision":        round(r["precision"], 4),
                "Rappel":           round(r["recall"], 4),
                "Temps entraîn.":   r["training_time_formatted"],
            } for r in results])

            st.dataframe(df_train, use_container_width=True, hide_index=True)

            # Radar chart
            st.divider()
            st.subheader("Radar chart — Kaggle T4")
            try:
                import plotly.graph_objects as go
                radar_m = ["mAP50", "mAP50-95", "Précision", "Rappel"]
                fig = go.Figure()
                colors_radar = ["#3498db", "#e74c3c", "#2ecc71"]
                for i, row in df_train.iterrows():
                    values = [float(row[m]) for m in radar_m]
                    values += [values[0]]
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_m + [radar_m[0]],
                        fill="toself",
                        name=row["Modèle"],
                        line_color=colors_radar[i % len(colors_radar)],
                        opacity=0.7,
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0.6, 0.85])),
                    showlegend=True,
                    title="Comparaison radar — entraînement Kaggle T4",
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                pass

            # Graphique barres
            st.divider()
            fig, ax = plt.subplots(figsize=(10, 4))
            metrics  = ["mAP50", "mAP50-95", "Précision", "Rappel"]
            x = np.arange(len(df_train))
            width = 0.8 / len(metrics)
            for i, metric in enumerate(metrics):
                ax.bar(x + i * width, df_train[metric], width, label=metric, alpha=0.85)
            ax.set_xticks(x + width * (len(metrics) - 1) / 2)
            ax.set_xticklabels(df_train["Modèle"])
            ax.set_ylim(0, 1.0)
            ax.legend(loc="lower right", fontsize=9)
            ax.set_title("Comparaison des métriques — entraînement Kaggle T4")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Analyse
            st.divider()
            st.subheader("Analyse et recommandation")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("""
                **YOLOv8n — Nano**
                - mAP50 : 77.6 %
                - Temps : 1h02 sur T4
                - ✅ Idéal pour la **démo temps réel** (webcam)
                - Léger (6 MB), rapide sur CPU/GPU
                """)
            with col_b:
                st.markdown("""
                **YOLOv8s — Small**
                - mAP50 : 77.7 %
                - Temps : 1h41 sur T4
                - ✅ **Meilleur compromis** précision / vitesse
                - Légèrement meilleure précision que Nano
                """)
            with col_c:
                st.markdown("""
                **YOLOv8m — Medium**
                - mAP50 : **78.3 %** 🏆
                - Temps : 3h27 sur T4
                - ✅ **Meilleure précision** pour l'analyse post-show
                - Plus lent — moins adapté au temps réel sans GPU
                """)

            best = max(results, key=lambda r: r["mAP50"])
            st.success(
                f"🏆 Meilleur modèle global : **{best['model_name']}** — mAP50 = {best['mAP50']:.4f}  \n"
                "Pour une démo temps réel (webcam), préférer **YOLOv8n** pour sa rapidité."
            )

    # Visualisations Kaggle par modèle
    _show_training_artifacts()

    # Limites
    st.divider()
    st.subheader("Limites et perspectives")
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.warning("""
        **Limites actuelles**
        - Précision ~68–78 % : des émotions subtiles (content vs neutral) sont difficiles à distinguer
        - Haar Cascade sensible à l'éclairage et aux angles de tête
        - Dataset principalement des visages frontaux, moins robuste de profil
        - Pas de gestion du mouvement (tracking inter-frames)
        """)
    with col_l2:
        st.info("""
        **Améliorations possibles**
        - Remplacer Haar Cascade par **RetinaFace** ou **YOLOv8-face** pour la détection
        - Entraîner sur un dataset plus diversifié (ethnicités, âges, éclairages)
        - Ajouter un **tracking** (DeepSORT) pour suivre les visages entre frames
        - Déploiement edge (ONNX, TensorRT) pour GPU embarqué en salle
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 5 — Démo
# ─────────────────────────────────────────────────────────────────────────────

def _infer_crowd(model, img_path: str, conf: float) -> list[dict]:
    """Pipeline 2 étapes : détecteur de visages OpenCV → crop → YOLO émotion sur chaque visage."""
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
    st.header("🎯 Démo — Analyse d'Émotion sur Image")

    st.markdown("""
    Upload une photo contenant un ou plusieurs visages. Le modèle détecte chaque visage,
    prédit l'émotion et calcule le score correspondant.

    > **Conseil :** Active le **Mode Foule** pour les photos avec plusieurs personnes.
    > Baisse le seuil de confiance si aucun visage n'est détecté (essaie 0.15).
    """)

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
                               help="OpenCV détecte chaque visage → YOLO sur chaque crop — idéal pour les groupes")

    if mode_foule:
        st.caption("ℹ️ Mode Foule : OpenCV Haar Cascade localise les visages → YOLOv8 classifie l'émotion sur chaque crop individuellement.")

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
        "Simule le score en temps réel en ajustant la distribution des émotions du public.  \n"
        "Utile pour comprendre comment le scoring réagit à différentes compositions de public."
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

        fractions = {e: v / total for e, v in sliders.items()}

    with col2:
        raw_score = sum(fractions[e] * EMOTION_WEIGHTS[e] for e in EMOTION_CLASSES)
        pct_score = _score_to_pct(raw_score)

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

        st.caption(
            f"Formule : raw = Σ(fraction_i × poids_i) = **{raw_score:+.3f}**  →  "
            f"score = ({raw_score:+.3f} + 1) / 2 × 100 = **{pct_score:.1f}%**"
        )

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

        axes[1].barh(["Score"], [pct_score], color=color, height=0.4)
        axes[1].barh(["Score"], [100 - pct_score], left=[pct_score], color="#ecf0f1", height=0.4)
        axes[1].set_xlim(0, 100)
        axes[1].axvline(50, color="gray", linestyle="--", alpha=0.5, label="Seuil neutre")
        axes[1].axvline(70, color="#2ecc71", linestyle="--", alpha=0.4, label="Seuil bon show")
        axes[1].set_xlabel("Score (%)")
        axes[1].set_title("Score humoriste")
        axes[1].legend(fontsize=8)
        axes[1].text(pct_score / 2, 0, f"{pct_score:.1f}%", ha="center", va="center",
                     fontweight="bold", color="white" if pct_score > 15 else "black")

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 7 — Caméra Live
# ─────────────────────────────────────────────────────────────────────────────

def _tab_camera() -> None:
    import cv2
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode

    st.header("📷 Caméra Live — Détection en Temps Réel")

    st.markdown("""
    **Pipeline en direct :**
    1. Chaque frame de la webcam est analysée par OpenCV Haar Cascade pour localiser les visages
    2. Chaque visage est cropé et passé dans le modèle YOLOv8 sélectionné
    3. Le score de la frame est affiché en overlay en haut à gauche
    """)

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

            for (bx1, by1, bx2, by2, emotion, conf_val) in overlay:
                color = COLORS_BGR.get(emotion, (128, 128, 128))
                cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
                label = f"{emotion} {conf_val:.0%}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(img, (bx1, by1 - th - 6), (bx1 + tw + 4, by1), color, -1)
                cv2.putText(img, label, (bx1 + 2, by1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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

    # ── Flux Temps Réel ───────────────────────────────────────────────────────
    st.subheader("🎥 Flux temps réel")
    st.info("Clique sur **START** pour activer la caméra. Les émotions s'affichent en overlay avec le score.")

    webrtc_streamer(
        key="emotion-live",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ── Snapshot ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📸 Snapshot — Prendre une photo")
    st.caption("Prends une photo avec ta webcam pour obtenir une analyse statique avec bounding boxes annotées.")

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
