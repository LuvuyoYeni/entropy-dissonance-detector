import random
from pathlib import Path

import nltk
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Ensure WordNet corpus is present (re-download if the zip is corrupt)
for _resource, _pkg in [
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]:
    try:
        nltk.data.find(_resource)
    except Exception:
        nltk.download(_pkg, quiet=False)

from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

SUBSTITUTION_RATE = 0.30
RANDOM_SEED = 42

# NLTK POS tags that map to WordNet POS constants
_WN_POS_MAP = {
    "NN": wordnet.NOUN,  "NNS": wordnet.NOUN,
    "NNP": wordnet.NOUN, "NNPS": wordnet.NOUN,
    "VB": wordnet.VERB,  "VBD": wordnet.VERB,
    "VBG": wordnet.VERB, "VBN": wordnet.VERB,
    "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
    "JJ": wordnet.ADJ,   "JJR": wordnet.ADJ,
    "JJS": wordnet.ADJ,
}


def _get_synonym(word, pos_tag):
    """Return the first WordNet synonym for word that differs from the original, or None."""
    wn_pos = _WN_POS_MAP.get(pos_tag)
    if wn_pos is None:
        return None
    for synset in wordnet.synsets(word.lower(), pos=wn_pos):
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and candidate.replace(" ", "").isalpha():
                return candidate.capitalize() if word[0].isupper() else candidate
    return None


def _obfuscate_sentence(sentence, rng):
    """Replace ~30 % of eligible content words with WordNet synonyms."""
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    eligible = [i for i, (w, p) in enumerate(tagged) if p in _WN_POS_MAP and w.isalpha()]
    n_replace = max(1, round(len(eligible) * SUBSTITUTION_RATE)) if eligible else 0
    to_replace = set(rng.sample(eligible, min(n_replace, len(eligible))))

    result = []
    for i, (word, pos) in enumerate(tagged):
        if i in to_replace:
            syn = _get_synonym(word, pos)
            result.append(syn if syn else word)
        else:
            result.append(word)
    return " ".join(result)


def _obfuscate_text(text, rng):
    """Obfuscate every sentence in a document."""
    return " ".join(_obfuscate_sentence(s, rng) for s in sent_tokenize(text))


def _plot_adversarial_comparison(d2_orig, d2_obf, d3_orig, d3_obf, plots_dir):
    labels = ["D2 Original", "D2 Obfuscated", "D3 Original", "D3 Obfuscated"]
    values = [d2_orig, d2_obf, d3_orig, d3_obf]
    colors = ["#7b9ecc", "#4a6fa5", "#e07b54", "#a84e2f"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(
        "Adversarial Robustness: AUROC Before and After Synonym Substitution",
        fontsize=11, fontweight="bold",
    )
    ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.6, label="Random baseline (0.5)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    plt.tight_layout()

    path = Path(plots_dir) / "adversarial_comparison.png"
    plt.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def run_adversarial_experiment(baseline_results, full_results, phishing_texts, plots_dir):
    """
    Apply synonym substitution to the phishing test split and re-score with
    the already-fitted D2 and D3 detectors.  Nothing is retrained.
    """
    print("\n=== Adversarial Experiment: Synonym Substitution ===")
    print(f"  Substitution rate : {SUBSTITUTION_RATE:.0%} of eligible content words per sentence")

    rng = random.Random(RANDOM_SEED)

    # Recover the same phishing test split used by both evaluate functions
    _, phish_test = train_test_split(phishing_texts, test_size=0.2, random_state=42)
    print(f"  Phishing test emails: {len(phish_test)}")

    print("  Applying synonym substitution …")
    obfuscated_phish = [_obfuscate_text(t, rng) for t in phish_test]

    # ------------------------------------------------------------------
    # D2 — BaselineDetector (H_lex only)
    # ------------------------------------------------------------------
    d2_detector = baseline_results["detector"]
    d2_norm_scores = list(baseline_results["norm_scores"])
    d2_orig_auroc = baseline_results["auroc"]

    d2_obf_phish_scores = d2_detector.score(obfuscated_phish)
    d2_labels_obf = [0] * len(d2_norm_scores) + [1] * len(d2_obf_phish_scores)
    d2_obf_auroc = roc_auc_score(d2_labels_obf, d2_norm_scores + d2_obf_phish_scores)
    d2_degradation = d2_orig_auroc - d2_obf_auroc

    # ------------------------------------------------------------------
    # D3 — JointEntropyDetector (Mahalanobis on H_lex + H_syn)
    # ------------------------------------------------------------------
    d3_detector = full_results["detector"]
    d3_norm_scores = list(full_results["norm_scores"])
    d3_orig_auroc = full_results["auroc"]

    d3_obf_phish_scores = d3_detector.score(obfuscated_phish)
    d3_labels_obf = [0] * len(d3_norm_scores) + [1] * len(d3_obf_phish_scores)
    d3_obf_auroc = roc_auc_score(d3_labels_obf, d3_norm_scores + d3_obf_phish_scores)
    d3_degradation = d3_orig_auroc - d3_obf_auroc

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  ADVERSARIAL EXPERIMENT RESULTS")
    print("=" * 62)
    print(f"  D2  H_lex only (baseline):")
    print(f"    AUROC original        : {d2_orig_auroc:.4f}")
    print(f"    AUROC after obfuscation: {d2_obf_auroc:.4f}")
    print(f"    Degradation           : {d2_degradation:+.4f}")
    print()
    print(f"  D3  Joint Mahalanobis (full system):")
    print(f"    AUROC original        : {d3_orig_auroc:.4f}")
    print(f"    AUROC after obfuscation: {d3_obf_auroc:.4f}")
    print(f"    Degradation           : {d3_degradation:+.4f}")
    print()
    if d2_degradation > d3_degradation:
        print("  Finding: D2 degraded MORE than D3 under synonym substitution.")
        print("  Lexical signals are more vulnerable to synonym obfuscation than")
        print("  syntactic structure — consistent with the stated hypothesis.")
    else:
        print("  Finding: D3 degraded as much or more than D2 under synonym substitution.")
        print("  This does not confirm the hypothesis — results reported as-is.")
    print("=" * 62)

    _plot_adversarial_comparison(d2_orig_auroc, d2_obf_auroc, d3_orig_auroc, d3_obf_auroc, plots_dir)

    return {
        "d2_orig_auroc": d2_orig_auroc,
        "d2_obf_auroc": d2_obf_auroc,
        "d2_degradation": d2_degradation,
        "d3_orig_auroc": d3_orig_auroc,
        "d3_obf_auroc": d3_obf_auroc,
        "d3_degradation": d3_degradation,
    }
