

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import build_language_model, extract_features

DATA_DIR = Path("data/processed")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


def manual_cov_2x2(X):
    """Same triple-loop covariance used in JointEntropyDetector.fit()."""
    mu = X.mean(axis=0)
    diff = X - mu
    cov = np.zeros((2, 2))
    for row in diff:
        for i in range(2):
            for j in range(2):
                cov[i][j] += row[i] * row[j]
    return cov / (len(X) - 1)


def report_correlation(h_lex, h_syn, label):
    n = len(h_lex)
    pearson_r, pearson_p = stats.pearsonr(h_lex, h_syn)
    spearman_rho, spearman_p = stats.spearmanr(h_lex, h_syn)
    print(f"\n--- {label} ---")
    print(f"  n                : {n} sentences")
    print(f"  H_lex  mean±std  : {np.mean(h_lex):.4f} ± {np.std(h_lex, ddof=1):.4f}")
    print(f"  H_syn  mean±std  : {np.mean(h_syn):.4f} ± {np.std(h_syn, ddof=1):.4f}")
    print(f"  Pearson  r       : {pearson_r:.4f}   (p={pearson_p:.4e})")
    print(f"  Spearman rho     : {spearman_rho:.4f}   (p={spearman_p:.4e})")
    return pearson_r, spearman_rho


def main():
    print("Loading data...")
    with open(DATA_DIR / "ham.json", encoding="utf-8") as f:
        ham = json.load(f)
    with open(DATA_DIR / "phishing.json", encoding="utf-8") as f:
        phish = json.load(f)
    print(f"  {len(ham)} ham,  {len(phish)} phishing")

    # --- exact same split as evaluate_full_system() ---
    norm_trainval, _norm_test = train_test_split(ham, test_size=0.2, random_state=42)
    norm_train, _norm_val     = train_test_split(norm_trainval, test_size=0.125, random_state=42)
    _, phish_test             = train_test_split(phish, test_size=0.2, random_state=42)
    print(f"  norm_train emails: {len(norm_train)}  |  phish_test emails: {len(phish_test)}")

    print("\nBuilding language model on norm_train (same as JointEntropyDetector.fit)...")
    lm = build_language_model(norm_train)

    print("Extracting features from norm_train (sentence-level)...")
    print("  Source: src/features.py :: extract_features(norm_train, lm)")
    norm_records = extract_features(norm_train, lm)
    norm_h_lex = np.array([r["h_lex"] for r in norm_records])
    norm_h_syn = np.array([r["h_syn"] for r in norm_records])

    print("Extracting features from phish_test (sentence-level)...")
    print("  Source: src/features.py :: extract_features(phish_test, lm)")
    phish_records = extract_features(phish_test, lm)
    phish_h_lex = np.array([r["h_lex"] for r in phish_records])
    phish_h_syn = np.array([r["h_syn"] for r in phish_records])

    # --- correlations ---
    norm_r, norm_rho   = report_correlation(norm_h_lex, norm_h_syn, "NORMAL TRAINING SET (Gaussian fit)")
    phish_r, phish_rho = report_correlation(phish_h_lex, phish_h_syn, "PHISHING TEST SET (held-out)")

    # --- covariance health (same manual computation as full_system.py) ---
    X = np.column_stack([norm_h_lex, norm_h_syn])
    cov = manual_cov_2x2(X)
    cond = np.linalg.cond(cov)
    det  = np.linalg.det(cov)
    print(f"\n--- COVARIANCE MATRIX (normal training set) ---")
    print(f"  [[{cov[0,0]:.6f}  {cov[0,1]:.6f}]")
    print(f"   [{cov[1,0]:.6f}  {cov[1,1]:.6f}]]")
    print(f"  Condition number : {cond:.2f}")
    print(f"  Determinant      : {det:.6f}")

    # --- scatter plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(norm_h_lex, norm_h_syn, alpha=0.15, s=6, color="#7b9ecc", label="Normal train", rasterized=True)
    ax.scatter(phish_h_lex, phish_h_syn, alpha=0.25, s=8, color="#e07b54", label="Phishing test", rasterized=True)
    # regression line on normal set
    m, b = np.polyfit(norm_h_lex, norm_h_syn, 1)
    x_line = np.linspace(norm_h_lex.min(), norm_h_lex.max(), 200)
    ax.plot(x_line, m * x_line + b, color="#334466", lw=1.0, linestyle="--", label="OLS (normal)")
    ax.set_xlabel("H_lex (mean per-token surprisal, bits)", fontsize=11)
    ax.set_ylabel("H_syn (dep-label entropy, bits)", fontsize=11)
    ax.set_title(f"H_lex vs H_syn  —  Pearson r = {norm_r:.3f}  (n={len(norm_h_lex)} sentences)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_correlation.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {out_path}")

    # --- interpretation ---
    abs_r = abs(norm_r)
    if abs_r >= 0.6:
        redundancy = "high (|r| >= 0.6)"
    elif abs_r >= 0.3:
        redundancy = f"moderate (0.3 <= |r| < 0.6)"
    else:
        redundancy = "low (|r| < 0.3)"

    if cond < 30:
        cov_health = f"well-conditioned (cond = {cond:.1f} < 30)"
    elif cond < 100:
        cov_health = f"borderline (cond = {cond:.1f}, 30 <= cond < 100)"
    else:
        cov_health = f"ill-conditioned (cond = {cond:.1f} >= 100)"

    # D3 explanation
    if abs_r >= 0.6:
        explanation = (
            f"H_lex and H_syn share {abs_r:.0%} linear variance (|r|={abs_r:.3f}), "
            f"so H_syn adds little independent discriminative signal. "
            f"The Mahalanobis distance effectively collapses toward the H_lex axis, "
            f"adding noise without benefit — consistent with D3 AUROC ({0.7136:.4f}) < D2 AUROC ({0.8087:.4f})."
        )
    elif abs_r >= 0.3:
        explanation = (
            f"Moderate correlation (|r|={abs_r:.3f}) means H_syn is partially redundant with H_lex. "
            f"H_syn also has low variance (std={np.std(norm_h_syn, ddof=1):.4f} bits), making it a weak discriminator. "
            f"The bivariate Gaussian adds a noisy second axis that degrades rather than improves separation "
            f"(D3 AUROC {0.7136:.4f} < D2 AUROC {0.8087:.4f}). "
            f"Condition number {cond:.1f} indicates the covariance is {'ill-conditioned' if cond >= 100 else 'borderline'}, "
            f"so Mahalanobis distance computation is numerically unstable."
        )
    else:
        explanation = (
            f"Low correlation (|r|={abs_r:.3f}) means H_syn and H_lex are largely independent. "
            f"The numbers alone do NOT explain D3 < D2. "
            f"Other candidates: H_syn has low discriminative power on its own (check D1 baseline); "
            f"covariance condition number {cond:.1f} may indicate numerical instability; "
            f"or phishing sentences do not lie outside the normal Gaussian in H_syn."
        )

    print("\n" + "=" * 62)
    print("  INTERPRETATION")
    print("=" * 62)
    print(f"FEATURE_REDUNDANCY:    {redundancy}")
    print(f"COVARIANCE_HEALTH:     {cov_health}")
    print(f"D3_LE_D2_EXPLANATION:  {explanation}")
    print("=" * 62)

    # flag if phishing correlation structure differs substantially
    diff = abs(phish_r - norm_r)
    if diff > 0.15:
        print(f"\n  FLAG: Pearson r differs by {diff:.3f} between normal ({norm_r:.3f}) and phishing ({phish_r:.3f}).")
        print("  This means the two classes occupy structurally different regions of feature space —")
        print("  a potentially useful finding for the paper.")
    else:
        print(f"\n  NOTE: Correlation structure is similar between normal ({norm_r:.3f}) and phishing ({phish_r:.3f}).")


if __name__ == "__main__":
    main()
