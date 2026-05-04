import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from src.features import sent_tokenize, build_language_model, compute_h_lex


class BaselineDetector:
    def __init__(self):
        self.language_model = None
        self.mean_ = None
        self.std_ = None
        self.threshold_ = None

    def fit(self, normal_texts, k=2.0):
        print("  Building n-gram language model on normal corpus...")
        self.language_model = build_language_model(normal_texts)

        scores = []
        for text in normal_texts:
            for sent in sent_tokenize(text):
                h = compute_h_lex(sent, self.language_model)
                if h is not None:
                    scores.append(h)

        scores = np.array(scores)
        self.mean_ = scores.mean()
        self.std_ = scores.std()
        self.threshold_ = self.mean_ + k * self.std_

        print(f"  H_lex stats on normal data  — mean: {self.mean_:.3f}  std: {self.std_:.3f}")
        print(f"  Anomaly threshold (mean + {k}*std): {self.threshold_:.3f}")
        return self

    def score(self, texts):
        scores = []
        for text in texts:
            for sent in sent_tokenize(text):
                h = compute_h_lex(sent, self.language_model)
                if h is not None:
                    scores.append(h)
        return scores

    def predict(self, texts):
        return [1 if s > self.threshold_ else 0 for s in self.score(texts)]


def evaluate_baseline(normal_texts, phishing_texts, k=2.0):
    print("\n=== D2 Baseline Evaluation (H_lex only) ===")

    norm_train, norm_test = train_test_split(normal_texts, test_size=0.2, random_state=42)
    _, phish_test = train_test_split(phishing_texts, test_size=0.2, random_state=42)

    print(f"  Train normal: {len(norm_train)}   Test normal: {len(norm_test)}   Test phishing: {len(phish_test)}")

    detector = BaselineDetector()
    detector.fit(norm_train, k=k)

    norm_scores = detector.score(norm_test)
    phish_scores = detector.score(phish_test)

    all_scores = norm_scores + phish_scores
    all_labels = [0] * len(norm_scores) + [1] * len(phish_scores)
    preds = [1 if s > detector.threshold_ else 0 for s in all_scores]
    auroc = roc_auc_score(all_labels, all_scores)

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
