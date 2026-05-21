# Entropy Dissonance Detector

I built this for my Honours NLP module at the University of Johannesburg to see whether phishing emails could be detected using entropy mismatch instead of standard keyword-heavy spam filtering. 

The idea is simple: phishing emails often sound "wrong" before they look wrong. Instead of relying only on suspicious words, I measure how statistically unusual a sentence is—both lexically and syntactically.

### How it works
Each sentence gets two signals:
* **$H_{lex}$**: Average token surprisal from a custom trigram Laplace language model.
* **$H_{syn}$**: Shannon entropy over spaCy dependency arc labels.

I fit a bivariate Gaussian on legitimate email, then score new sentences using Mahalanobis distance. If a sentence drifts too far from the “normal email” distribution, the system flags it as phishing.

**There are two detectors in the repo:**
1. **D2:** Baseline using only $H_{lex}$.
2. **D3:** Full system using joint Mahalanobis distance on ($H_{lex}$, $H_{syn}$).

I also included an adversarial evaluation that attacks the lexical signal using WordNet synonym substitution to see how robust each detector actually is.

---

### Project Structure

The project is organized into a few core scripts and a source directory for the logic:

* **src/**: Contains the engine of the project, including `features.py` (trigram model and entropy calculations), `baseline.py`, `full_system.py`, and the `adversarial.py` experiment.
* **download_data.py**: Handles downloading and parsing the SpamAssassin and Nazario corpora.
* **evaluate.py**: The main entry point. This runs the baseline, full system, and adversarial tests in sequence.
* **requirements.txt**: List of necessary libraries.

---

### Getting Started

**Prerequisites:** Requires **Python 3.11**. 

1. **Setup environment:**
   ```bash
   python3.11 -m venv venv311
   source venv311/bin/activate  # macOS / Linux
   # venv311\Scripts\activate   # Windows