import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve
from src.baseline import evaluate_baseline
from src.full_system import evaluate_full_system
from src.adversarial import run_adversarial_experiment

DATA_DIR = Path("data/processed")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load_data():
    ham_path = DATA_DIR / "ham.json"
    phish_path = DATA_DIR / "phishing.json"

    if not ham_path.exists() or not phish_path.exists():
        raise FileNotFoundError("Processed data not found. Run: python download_data.py")

    with open(ham_path, encoding="utf-8") as f:
        ham = json.load(f)
    with open(phish_path, encoding="utf-8") as f:
        phish = json.load(f)

    print(f"Loaded {len(ham)} legitimate and {len(phish)} phishing emails")
    return ham, phish


def plot_roc_comparison(baseline_results, full_results):
    _, ax = plt.subplots(figsize=(7, 6))

    for results, label, color in [
        (baseline_results, "Baseline (H_lex only)", "#7b9ecc"),
        (full_results, "Full System (Joint Mahalanobis)", "#e07b54"),
    ]:
        fpr, tpr, _ = roc_curve(results["all_labels"], results["all_scores"])
        ax.plot(fpr, tpr, label=f"{label}  (AUROC = {results['auroc']:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    path = PLOTS_DIR / "roc_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def plot_score_distributions(baseline_results, full_results):
    _, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(baseline_results["norm_scores"], bins=40, alpha=0.6, color="#7b9ecc", label="Normal")
    axes[0].hist(baseline_results["phish_scores"], bins=40, alpha=0.6, color="#e07b54", label="Phishing")
    axes[0].set_title("D2 Baseline: H_lex Distribution", fontweight="bold")
    axes[0].set_xlabel("H_lex (mean surprisal)")
    axes[0].set_ylabel("Sentence count")
    axes[0].legend()

    axes[1].hist(full_results["norm_scores"], bins=40, alpha=0.6, color="#7b9ecc", label="Normal")
    axes[1].hist(full_results["phish_scores"], bins=40, alpha=0.6, color="#e07b54", label="Phishing")
    axes[1].set_title("D3 Full System: Mahalanobis Distance Distribution", fontweight="bold")
    axes[1].set_xlabel("Mahalanobis Distance D")
    axes[1].set_ylabel("Sentence count")
    axes[1].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax)

    plt.tight_layout()
    path = PLOTS_DIR / "score_distributions.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def plot_feature_space(full_results):
    from src.features import extract_features
    from matplotlib.patches import Ellipse

    detector = full_results["detector"]

    with open(DATA_DIR / "ham.json", encoding="utf-8") as f:
        ham = json.load(f)[:200]
    with open(DATA_DIR / "phishing.json", encoding="utf-8") as f:
        phish = json.load(f)[:200]

    norm_rec = extract_features(ham, detector.language_model)
    phish_rec = extract_features(phish, detector.language_model)

    norm_x = [r["h_lex"] for r in norm_rec]
    norm_y = [r["h_syn"] for r in norm_rec]
    phish_x = [r["h_lex"] for r in phish_rec]
    phish_y = [r["h_syn"] for r in phish_rec]

    _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(norm_x, norm_y, alpha=0.35, s=12, color="#7b9ecc", label="Normal", zorder=2)
    ax.scatter(phish_x, phish_y, alpha=0.35, s=12, color="#e07b54", label="Phishing", zorder=2)

    # recover covariance from sigma_inv_ to draw the Mahalanobis decision ellipse
    cov = np.linalg.inv(detector.sigma_inv_)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * detector.threshold_ * np.sqrt(vals)
    ellipse = Ellipse(
        xy=detector.mu_, width=w, height=h, angle=theta,
        edgecolor="#333333", fc="none", lw=1.5, linestyle="--",
        label=f"Decision boundary (tau={detector.threshold_:.2f})"
    )
    ax.add_patch(ellipse)
    ax.plot(*detector.mu_, "k+", ms=10, mew=2, label="Normal centroid (mu)")

    ax.set_xlabel("H_lex (lexical surprisal)", fontsize=12)
    ax.set_ylabel("H_syn (syntactic entropy)", fontsize=12)
    ax.set_title("Joint Feature Space: Normal vs Phishing", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    sns.despine()
    plt.tight_layout()
    path = PLOTS_DIR / "feature_space.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def print_summary(baseline_results, full_results):
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"  D2 Baseline AUROC   : {baseline_results['auroc']:.4f}")
    print(f"  D3 Full System AUROC: {full_results['auroc']:.4f}")
    improvement = full_results["auroc"] - baseline_results["auroc"]
    print(f"  Improvement         : {improvement:+.4f}")
    print("=" * 50)


def main():
    print("Loading data...")
    ham, phish = load_data()

    baseline_results = evaluate_baseline(ham, phish)
    full_results = evaluate_full_system(ham, phish)

    print("\nGenerating plots...")
    plot_roc_comparison(baseline_results, full_results)
    plot_score_distributions(baseline_results, full_results)
    plot_feature_space(full_results)

    print_summary(baseline_results, full_results)

    run_adversarial_experiment(baseline_results, full_results, phish, PLOTS_DIR)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
