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
    MODELS_DIR,
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
        st.info(
            "Les résultats ne sont pas encore disponibles.  \n"
            "Lance `python scripts/main.py` une fois les modèles `.pt` placés dans `models/` "
            "pour générer `results/model_metrics.csv`."
        )
        # Tableau de résultats attendus (estimations benchmark)
        st.subheader("Performances attendues (estimations)")
        expected = pd.DataFrame([
            {"Modèle": "YOLOv8n", "mAP50 (attendu)": "~0.65", "Inférence": "~2 ms/img",  "Taille": "6 MB"},
            {"Modèle": "YOLOv8s", "mAP50 (attendu)": "~0.72", "Inférence": "~4 ms/img",  "Taille": "22 MB"},
            {"Modèle": "YOLOv8m", "mAP50 (attendu)": "~0.76", "Inférence": "~8 ms/img",  "Taille": "52 MB"},
        ])
        st.dataframe(expected, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 5 — Démo
# ─────────────────────────────────────────────────────────────────────────────

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

    model_key = st.selectbox(
        "Modèle",
        options=list(available.keys()),
        format_func=lambda k: available[k]["name"],
    )
    conf = st.slider("Seuil de confiance", 0.1, 0.9, 0.35, 0.05)

    uploaded = st.file_uploader("Charge une image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded:
        import tempfile
        from model_io import load_model

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
                    result = model.model(tmp_path, conf=conf, verbose=False)[0]
                    boxes  = result.boxes

                    if boxes is None or len(boxes) == 0:
                        st.warning("Aucun visage détecté. Essaie avec une autre image ou baisse le seuil.")
                    else:
                        st.success(f"{len(boxes)} visage(s) détecté(s)")
                        for i, box in enumerate(boxes):
                            cls_id   = int(box.cls[0])
                            emotion  = EMOTION_CLASSES[cls_id]
                            conf_val = float(box.conf[0])
                            weight   = EMOTION_WEIGHTS[emotion]
                            score    = _score_to_pct(weight)
                            r, g, b  = EMOTION_COLORS[emotion]
                            st.markdown(
                                f"<div style='border-left:4px solid rgb({r},{g},{b});padding:8px;margin:4px 0'>"
                                f"<b>Visage {i+1}</b> — {emotion.upper()} "
                                f"({conf_val:.0%} confiance)<br>"
                                f"Contribution au score : <b>{score:.0f}%</b></div>",
                                unsafe_allow_html=True,
                            )

                        # Score global si plusieurs visages
                        if len(boxes) > 1:
                            emotions = [EMOTION_CLASSES[int(b.cls[0])] for b in boxes]
                            scores   = [_score_to_pct(EMOTION_WEIGHTS[e]) for e in emotions]
                            global_s = float(np.mean(scores))
                            st.metric("Score global de cette frame", f"{global_s:.1f}%")
                except Exception as e:
                    st.error(f"Erreur lors de l'inférence : {e}")

    st.divider()
    st.subheader("Tester en temps réel (webcam)")
    best = next(iter(available.values())) if available else None
    if best:
        st.code(
            f"python emotion_scorer/predict_emotions.py \\\n"
            f"  --model {Path(best['path']).relative_to(Path(best['path']).parent.parent)} \\\n"
            f"  --source 0",
            language="bash",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Onglet 6 — Score Live (simulation)
# ─────────────────────────────────────────────────────────────────────────────

def _tab_score() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    st.header("🎤 Simulateur de Score Humoriste")
    st.markdown(
        "Simule le score en temps réel en ajustant la distribution des émotions du public."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Distribution du public")
        sliders = {}
        remaining = 100
        for i, emotion in enumerate(EMOTION_CLASSES):
            max_val = max(0, remaining) if i < len(EMOTION_CLASSES) - 1 else remaining
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
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> None:
    st.set_page_config(
        page_title="Comedian Emotion Scorer",
        page_icon="🎤",
        layout="wide",
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Présentation",
        "📊 Dataset",
        "🤖 Modèles",
        "📈 Résultats",
        "🎯 Démo",
        "🎤 Score Live",
    ])

    with tab1: _tab_presentation()
    with tab2: _tab_dataset()
    with tab3: _tab_models()
    with tab4: _tab_results()
    with tab5: _tab_demo()
    with tab6: _tab_score()


if __name__ == "__main__":
    build_app()
