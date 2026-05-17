"""
Script de comparaison des modèles entraînés pour la détection d'émotions faciales
Génère un rapport visuel et textuel des performances de chaque modèle
"""

import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def load_results(results_path: str = "training_results.json") -> list:
    """Charge les résultats d'entraînement depuis le fichier JSON"""
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Fichier de résultats non trouvé: {results_path}\n"
            "Exécutez d'abord: python train_models.py"
        )
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison_table(results: list):
    """Affiche un tableau comparatif dans le terminal"""
    print("\n" + "="*80)
    print("COMPARAISON DES MODÈLES - DÉTECTION D'ÉMOTIONS FACIALES")
    print("="*80)
    
    # Filtrer les résultats valides
    valid_results = [r for r in results if "error" not in r]
    
    if not valid_results:
        print("Aucun résultat valide trouvé.")
        return
    
    # Trier par mAP50-95 décroissant
    valid_results.sort(key=lambda x: x.get("mAP50_95", 0), reverse=True)
    
    # Header
    print(f"{'Modèle':<15} {'mAP50':>8} {'mAP50-95':>10} {'Précision':>10} {'Rappel':>8} {'Temps':>12}")
    print("-"*80)
    
    for i, r in enumerate(valid_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(
            f"{rank} {r['model_name']:<13} "
            f"{r.get('mAP50', 0):>8.4f} "
            f"{r.get('mAP50_95', 0):>10.4f} "
            f"{r.get('precision', 0):>10.4f} "
            f"{r.get('recall', 0):>8.4f} "
            f"{r.get('training_time_formatted', 'N/A'):>12}"
        )
    
    print("-"*80)
    
    # Meilleur modèle
    best = valid_results[0]
    print(f"\n🏆 MEILLEUR MODÈLE: {best['model_name']}")
    print(f"   mAP50: {best.get('mAP50', 0):.4f} | mAP50-95: {best.get('mAP50_95', 0):.4f}")
    print(f"   Poids: {best.get('best_model_path', 'N/A')}")


def plot_comparison(results: list, save_path: str = "model_comparison.png"):
    """Génère un graphique comparatif des modèles"""
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("Aucun résultat valide pour le graphique.")
        return
    
    valid_results.sort(key=lambda x: x.get("mAP50_95", 0), reverse=True)
    
    model_names = [r["model_name"] for r in valid_results]
    metrics = {
        "mAP50": [r.get("mAP50", 0) for r in valid_results],
        "mAP50-95": [r.get("mAP50_95", 0) for r in valid_results],
        "Précision": [r.get("precision", 0) for r in valid_results],
        "Rappel": [r.get("recall", 0) for r in valid_results],
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Comparaison des Modèles - Détection d'Émotions Faciales", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    # Graphique barres groupées
    x = np.arange(len(model_names))
    width = 0.2
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    
    ax1 = axes[0]
    for i, (metric_name, values) in enumerate(metrics.items()):
        bars = ax1.bar(x + i * width, values, width, label=metric_name, color=colors[i], alpha=0.85)
    
    ax1.set_xlabel("Modèles", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_title("Métriques par Modèle", fontsize=13)
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    
    # Graphique radar (si assez de modèles)
    ax2 = axes[1]
    categories = list(metrics.keys())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax2 = plt.subplot(122, polar=True)
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=10)
    ax2.set_ylim(0, 1)
    ax2.set_title("Radar des Performances", fontsize=13, pad=20)
    
    radar_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for i, r in enumerate(valid_results):
        values_radar = [r.get("mAP50", 0), r.get("mAP50_95", 0), 
                        r.get("precision", 0), r.get("recall", 0)]
        values_radar += values_radar[:1]
        color = radar_colors[i % len(radar_colors)]
        ax2.plot(angles, values_radar, "o-", linewidth=2, label=r["model_name"], color=color)
        ax2.fill(angles, values_radar, alpha=0.1, color=color)
    
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n📈 Graphique sauvegardé: {save_path}")
    plt.show()


def evaluate_on_test_set(results: list, dataset_yaml: str = "Human-face-emotions-1/data.yaml"):
    """Évalue les modèles entraînés sur le jeu de test"""
    print("\n" + "="*60)
    print("ÉVALUATION SUR LE JEU DE TEST")
    print("="*60)
    
    test_results = []
    valid_results = [r for r in results if "error" not in r and "best_model_path" in r]
    
    for r in valid_results:
        model_path = r["best_model_path"]
        if not os.path.exists(model_path):
            print(f"⚠️  Modèle non trouvé: {model_path}")
            continue
        
        print(f"\nÉvaluation de {r['model_name']}...")
        model = YOLO(model_path)
        
        # Évaluation sur le jeu de test
        start = time.time()
        eval_results = model.val(data=dataset_yaml, split="test")
        inference_time = time.time() - start
        
        test_metrics = {
            "model_name": r["model_name"],
            "test_mAP50": float(eval_results.results_dict.get("metrics/mAP50(B)", 0)),
            "test_mAP50_95": float(eval_results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "inference_time_ms": round(inference_time * 1000 / 941, 2),  # ms par image
        }
        test_results.append(test_metrics)
        print(f"  Test mAP50: {test_metrics['test_mAP50']:.4f}")
        print(f"  Test mAP50-95: {test_metrics['test_mAP50_95']:.4f}")
        print(f"  Inférence: {test_metrics['inference_time_ms']:.2f} ms/image")
    
    return test_results


def generate_report(results: list, test_results: list = None):
    """Génère un rapport Markdown complet"""
    report = []
    report.append("# Rapport de Comparaison des Modèles")
    report.append("## Détection d'Émotions Faciales - Human Face Emotions Dataset")
    report.append("")
    report.append("### Informations sur le Dataset")
    report.append("| Paramètre | Valeur |")
    report.append("|-----------|--------|")
    report.append("| Nombre d'images | 9 400 |")
    report.append("| Nombre de classes | 8 |")
    report.append("| Classes | anger, content, disgust, fear, happy, neutral, sad, surprise |")
    report.append("| Train/Val/Test | 70% / 20% / 10% |")
    report.append("| Format | YOLOv8 (TXT + YAML) |")
    report.append("")
    
    valid_results = [r for r in results if "error" not in r]
    valid_results.sort(key=lambda x: x.get("mAP50_95", 0), reverse=True)
    
    report.append("### Résultats d'Entraînement")
    report.append("")
    report.append("| Rang | Modèle | mAP50 | mAP50-95 | Précision | Rappel | Temps |")
    report.append("|------|--------|-------|----------|-----------|--------|-------|")
    
    medals = ["🥇", "🥈", "🥉"]
    for i, r in enumerate(valid_results):
        medal = medals[i] if i < 3 else str(i+1)
        report.append(
            f"| {medal} | {r['model_name']} | "
            f"{r.get('mAP50', 0):.4f} | {r.get('mAP50_95', 0):.4f} | "
            f"{r.get('precision', 0):.4f} | {r.get('recall', 0):.4f} | "
            f"{r.get('training_time_formatted', 'N/A')} |"
        )
    
    if test_results:
        report.append("")
        report.append("### Résultats sur le Jeu de Test")
        report.append("")
        report.append("| Modèle | Test mAP50 | Test mAP50-95 | Inférence (ms/img) |")
        report.append("|--------|-----------|--------------|-------------------|")
        for t in test_results:
            report.append(
                f"| {t['model_name']} | {t.get('test_mAP50', 0):.4f} | "
                f"{t.get('test_mAP50_95', 0):.4f} | {t.get('inference_time_ms', 0):.2f} |"
            )
    
    if valid_results:
        best = valid_results[0]
        report.append("")
        report.append(f"### 🏆 Modèle Recommandé: **{best['model_name']}**")
        report.append("")
        report.append(f"- **mAP50**: {best.get('mAP50', 0):.4f}")
        report.append(f"- **mAP50-95**: {best.get('mAP50_95', 0):.4f}")
        report.append(f"- **Poids**: `{best.get('best_model_path', 'N/A')}`")
        report.append("")
        report.append("### Application: Notation d'Humoriste")
        report.append("")
        report.append("Pour scorer un humoriste basé sur les émotions des spectateurs:")
        report.append("- **happy** + **content** = émotions positives (score +)")
        report.append("- **neutral** = émotions neutres (score 0)")
        report.append("- **anger** + **disgust** + **sad** = émotions négatives (score -)")
        report.append("- **surprise** = impact fort (score variable)")
        report.append("")
        report.append("Score = (happy + content - anger - disgust - sad + 0.5*surprise) / total_détections")
    
    report_text = "\n".join(report)
    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print("\n📋 Rapport sauvegardé: RESULTS.md")
    return report_text


if __name__ == "__main__":
    # Charger les résultats
    results = load_results()
    
    # Afficher la comparaison dans le terminal
    print_comparison_table(results)
    
    # Générer le graphique
    plot_comparison(results)
    
    # Générer le rapport Markdown
    generate_report(results)
    
    print("\n✅ Comparaison terminée! Consultez RESULTS.md et model_comparison.png")
