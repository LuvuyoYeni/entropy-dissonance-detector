"""
D1 baseline: H_syn only, univariate Gaussian anomaly detector.

Mirrors D2 (src/baseline.py) structure exactly. Differences:
  - feature   : H_syn (dep-label entropy via spaCy, no language model)
  - score     : ((h_syn - mu) / sigma)**2  — 1-D Mahalanobis squared
  - threshold : k**2  (equivalent to k standard deviations)
"""

import csv
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import sent_tokenize, compute_h_syn
from src.baseline import evaluate_baseline
from src.full_system import evaluate_full_system
from src.adversarial import _obfuscate_text, SUBSTITUTION_RATE, RANDOM_SEED

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class D1SynDetector:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.threshold_ = None

    def fit(self, normal_texts, k=2.0):
        scores = []
        for text in normal_texts:
            for sent in sent_tokenize(text):
                h = compute_h_syn(sent)
                if h is not None:
                    scores.append(h)
        arr = np.array(scores)
        self.mean_ = arr.mean()
        self.std_ = arr.std()
        self.threshold_ = k ** 2
        print(f"  H_syn stats on normal data  — mean: {self.mean_:.4f}  std: {self.std_:.4f}")
        print(f"  Anomaly threshold (k={k}, threshold=k^2): {self.threshold_:.4f}")
        return self

    def score(self, texts):
        scores = []
        for text in texts:
            for sent in sent_tokenize(text):
                h = compute_h_syn(sent)
                if h is not None:
                    scores.append(((h - self.mean_) / self.std_) ** 2)
        return scores

    def predict(self, texts):
        return [1 if s > self.threshold_ else 0 for s in self.score(texts)]


def evaluate_d1(normal_texts, phishing_texts, k=2.0):
    print("\n=== D1 Baseline Evaluation (H_syn only) ===")

    norm_train, norm_test = train_test_split(normal_texts, test_size=0.2, random_state=42)
    _, phish_test = train_test_split(phishing_texts, test_size=0.2, random_state=42)

    print(f"  Train normal: {len(norm_train)}   Test normal: {len(norm_test)}   Test phishing: {len(phish_test)}")

    detector = D1SynDetector()
    detector.fit(norm_train, k=k)

    norm_scores = detector.score(norm_test)
    phish_scores = detector.score(phish_test)

    all_scores = norm_scores + phish_scores
    all_labels = [0] * len(norm_scores) + [1] * len(phish_scores)
    preds = [1 if s > detector.threshold_ else 0 for s in all_scores]
    auroc = roc_auc_score(all_labels, all_scores)

    print(f"  Test sentences — normal: {len(norm_scores)}   phishing: {len(phish_scores)}")
    print(f"\n  AUROC: {auroc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_labels, preds, target_names=["Normal", "Phishing"]))

    return {
        "detector": detector,
        "auroc": auroc,
        "all_scores": all_scores,
        "all_labels": all_labels,
        "norm_scores": norm_scores,
        "phish_scores": phish_scores,
    }


def _adversarial_auroc(detector, norm_scores, obf_phish):
    obf_scores = detector.score(obf_phish)
    labels = [0] * len(norm_scores) + [1] * len(obf_scores)
    return roc_auc_score(labels, norm_scores + obf_scores)


def main():
    print("Loading data...")
    with open("data/processed/ham.json", encoding="utf-8") as f:
        ham = json.load(f)
    with open("data/processed/phishing.json", encoding="utf-8") as f:
        phish = json.load(f)
    print(f"  {len(ham)} ham,  {len(phish)} phishing")

    d1_results = evaluate_d1(ham, phish)
    d2_results = evaluate_baseline(ham, phish)
    d3_results = evaluate_full_system(ham, phish)

    
    D2_EXPECTED, D3_EXPECTED = 0.8087, 0.7136
    print("\n--- Reproducibility check ---")
    d2_ok = abs(d2_results["auroc"] - D2_EXPECTED) < 0.0005
    d3_ok = abs(d3_results["auroc"] - D3_EXPECTED) < 0.0005
    print(f"  D2 AUROC: {d2_results['auroc']:.4f}  expected {D2_EXPECTED}  {'OK' if d2_ok else 'MISMATCH'}")
    print(f"  D3 AUROC: {d3_results['auroc']:.4f}  expected {D3_EXPECTED}  {'OK' if d3_ok else 'MISMATCH'}")
    if not d2_ok or not d3_ok:
        print("  STOPPING: clean-AUROC reproducibility failed.")
        sys.exit(1)

   
    print(f"\n=== Adversarial: synonym substitution (rate {SUBSTITUTION_RATE:.0%}) ===")
    rng = random.Random(RANDOM_SEED)
    _, phish_test = train_test_split(phish, test_size=0.2, random_state=42)
    print(f"  Phishing test emails: {len(phish_test)}")
    print("  Applying synonym substitution...")
    obf_phish = [_obfuscate_text(t, rng) for t in phish_test]

    d1_adv = _adversarial_auroc(d1_results["detector"], d1_results["norm_scores"], obf_phish)
    d2_adv = _adversarial_auroc(d2_results["detector"], d2_results["norm_scores"], obf_phish)
    d3_adv = _adversarial_auroc(d3_results["detector"], d3_results["norm_scores"], obf_phish)

    
    D2_ADV_EXPECTED, D3_ADV_EXPECTED = 0.8561, 0.7789
    print("\n--- Adversarial reproducibility check ---")
    d2_adv_ok = abs(d2_adv - D2_ADV_EXPECTED) < 0.0005
    d3_adv_ok = abs(d3_adv - D3_ADV_EXPECTED) < 0.0005
    print(f"  D2 adv AUROC: {d2_adv:.4f}  expected {D2_ADV_EXPECTED}  {'OK' if d2_adv_ok else 'MISMATCH'}")
    print(f"  D3 adv AUROC: {d3_adv:.4f}  expected {D3_ADV_EXPECTED}  {'OK' if d3_adv_ok else 'MISMATCH'}")
    if not d2_adv_ok or not d3_adv_ok:
        print("  STOPPING: adversarial-AUROC reproducibility failed.")
        sys.exit(1)

    d1_clean = d1_results["auroc"]
    d2_clean = d2_results["auroc"]
    d3_clean = d3_results["auroc"]

    
    csv_path = RESULTS_DIR / "three_way_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "features", "auroc_clean", "auroc_adversarial", "delta"])
        writer.writerow(["D1", "H_syn",      f"{d1_clean:.4f}", f"{d1_adv:.4f}", f"{d1_clean - d1_adv:+.4f}"])
        writer.writerow(["D2", "H_lex",      f"{d2_clean:.4f}", f"{d2_adv:.4f}", f"{d2_clean - d2_adv:+.4f}"])
        writer.writerow(["D3", "H_lex+H_syn",f"{d3_clean:.4f}", f"{d3_adv:.4f}", f"{d3_clean - d3_adv:+.4f}"])
    print(f"\n  Saved: {csv_path}")

   
    print("\n" + "=" * 66)
    print("  THREE-WAY COMPARISON")
    print("=" * 66)
    print(f"  {'Det':<5} {'Features':<14} {'Clean':>10} {'Adversarial':>13} {'Delta':>8}")
    print(f"  {'-'*54}")
    for det, feat, clean, adv in [
        ("D1", "H_syn",       d1_clean, d1_adv),
        ("D2", "H_lex",       d2_clean, d2_adv),
        ("D3", "H_lex+H_syn", d3_clean, d3_adv),
    ]:
        print(f"  {det:<5} {feat:<14} {clean:>10.4f} {adv:>13.4f} {clean-adv:>+8.4f}")
    print("=" * 66)

    
    if 0.50 <= d1_clean <= 0.55:
        prediction = "confirmed (0.50 <= AUROC <= 0.55)"
        implication = (
            "H_syn alone is near chance. The joint model's underperformance relative to D2 "
            "is fully explained by H_syn contributing noise: the feature carries almost no "
            "discriminative signal on its own, so adding it to the Mahalanobis distance "
            "dilutes the H_lex signal without benefit."
        )
    elif 0.55 < d1_clean <= 0.65:
        prediction = "weak_signal (0.55 < AUROC <= 0.65)"
        implication = (
            f"H_syn carries marginal discriminative information (AUROC {d1_clean:.4f}). "
            "The picture is more nuanced than the correlation analysis suggested — H_syn alone "
            "is weak but non-trivial. The joint model should in principle benefit from H_syn, "
            "but the Mahalanobis combination failed to exploit this, possibly because the "
            "moderate H_lex/H_syn correlation (r=0.44) absorbs the independent H_syn signal."
        )
    else:
        prediction = f"unexpected (AUROC = {d1_clean:.4f}, outside [0.50, 0.65])"
        implication = (
            "The diagnosis from the correlation analysis is incomplete. "
            f"D1 AUROC {d1_clean:.4f} is outside the expected range. "
            "Next steps: (1) inspect the H_syn score distribution for normal vs phishing; "
            "(2) verify sentence lengths are comparable across classes; "
            "(3) check whether spaCy parse quality differs between ham and phishing text."
        )

    print("\n" + "=" * 66)
    print("  INTERPRETATION")
    print("=" * 66)
    print(f"D1_AUROC:         {d1_clean:.4f}")
    print(f"PREDICTION_MATCH: {prediction}")
    print(f"IMPLICATION:      {implication}")
    print("=" * 66)


if __name__ == "__main__":
    main()
