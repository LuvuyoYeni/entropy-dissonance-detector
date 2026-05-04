# Entropy Dissonance Detector

Phishing email detection using bivariate entropy anomaly scoring, built for the Honours NLP module at the University of Johannesburg. Two entropy signals are computed per sentence: **H_lex** (mean per-token surprisal under a custom trigram Laplace language model) and **H_syn** (Shannon entropy over spaCy dependency arc labels). A bivariate Gaussian is fitted on legitimate email; new sentences are scored by their Mahalanobis distance from that distribution. Sentences that are jointly anomalous in both signals are flagged as phishing. An adversarial evaluation is also included, which applies WordNet synonym substitution to phishing test sentences and measures how much each detector's AUROC degrades.

Two systems are compared:

- **D2 (baseline)** — H_lex threshold only
- **D3 (full system)** — joint Mahalanobis distance on (H_lex, H_syn)

## Project structure

```
entropy-dissonance-detector/
├── src/
│   ├── features.py       # custom sentence/word tokenisers, trigram LM, H_lex, H_syn
│   ├── baseline.py       # D2 detector and evaluation
│   ├── full_system.py    # D3 detector and evaluation
│   └── adversarial.py    # synonym-substitution robustness experiment
├── download_data.py      # downloads and parses both corpora
├── evaluate.py           # runs all three experiments, prints results, saves plots
└── requirements.txt
```

## Setup

Requires **Python 3.11**. Python 3.13 is not supported (spaCy binary wheels are unavailable for it).

```bash
# create and activate a virtual environment
python3.11 -m venv venv311
source venv311/bin/activate        # macOS / Linux
# venv311\Scripts\activate         # Windows

# install dependencies
pip install -r requirements.txt

# download the spaCy English model
python -m spacy download en_core_web_sm
```

## Download the data

Run once before anything else:

```bash
python download_data.py
```

This fetches:
- SpamAssassin `easy_ham` + `easy_ham_2` (~3 900 legitimate emails)
- Nazario `phishing3.mbox` (~1 300 phishing emails)

Processed data is saved to `data/processed/ham.json` and `data/processed/phishing.json`. The raw archives are saved to `data/raw/` and neither directory is tracked by git.

## Run the system

```bash
python evaluate.py
```

This runs all three stages in sequence:

1. D2 baseline evaluation (H_lex only)
2. D3 full system evaluation (joint Mahalanobis)
3. Adversarial experiment (synonym substitution on phishing test set)

## Expected output

Console output shows AUROC scores and classification reports for D2 and D3, followed by the adversarial robustness summary. Four plots are saved to `plots/`:

| File | Contents |
|------|----------|
| `roc_comparison.png` | ROC curves for D2 and D3 |
| `score_distributions.png` | H_lex and Mahalanobis score histograms |
| `feature_space.png` | 2-D scatter of (H_lex, H_syn) with decision ellipse |
| `adversarial_comparison.png` | Bar chart of AUROC before/after obfuscation |

Approximate AUROC results:

| System | AUROC |
|--------|-------|
| D2 Baseline (H_lex only) | 0.77 |
| D3 Full System (Joint Mahalanobis) | 0.68 |

D2 outperforms D3 because H_syn does not separate the classes well on the Nazario corpus — phishing emails vary their vocabulary more than their grammatical structure, so the lexical signal carries most of the discriminative power. The adversarial experiment tests whether synonym substitution degrades D2 more than D3, since replacing words attacks the lexical signal while leaving syntactic structure intact.

## Module

IT08X87 — Natural Language Processing, University of Johannesburg
