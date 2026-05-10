import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split
from src.features import build_language_model, extract_features


class JointEntropyDetector:
    def __init__(self):
        self.language_model = None
        self.mu_ = None
        self.sigma_inv_ = None
        self.threshold_ = None

    def fit(self, normal_texts, val_texts=None, val_labels=None):
        print("  Building n-gram language model on normal corpus...")
        self.language_model = build_language_model(normal_texts)

        print("  Extracting (H_lex, H_syn) features from normal corpus...")
        records = extract_features(normal_texts, self.language_model)
        X = np.array([[r["h_lex"], r["h_syn"]] for r in records])

        self.mu_ = X.mean(axis=0)

        n = len(X)
        diff = X - self.mu_
        cov = np.zeros((2, 2))
        for row in diff:
            for i in range(2):
                for j in range(2):
                    cov[i][j] += row[i] * row[j]
        cov = cov / (n - 1)

        try:
            self.sigma_inv_ = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            self.sigma_inv_ = np.linalg.pinv(cov)

        print(f"  mu = [H_lex: {self.mu_[0]:.3f}, H_syn: {self.mu_[1]:.3f}]")
        print(f"  Covariance matrix:\n{cov.round(4)}")

        normal_distances = [self._mahalanobis(r["h_lex"], r["h_syn"]) for r in records]
        percentile_tau = float(np.percentile(normal_distances, 95))

        if val_texts is not None and val_labels is not None:
            sent_scores = []
            sent_labels = []
            for doc_text, label in zip(val_texts, val_labels):
                for r in extract_features([doc_text], self.language_model):
                    sent_scores.append(self._mahalanobis(r["h_lex"], r["h_syn"]))
                    sent_labels.append(label)

            if len(sent_scores) == 0 or len(set(sent_labels)) < 2:
                self.threshold_ = percentile_tau
            else:
                _, _, thresholds = roc_curve(sent_labels, sent_scores)
                f1_scores = []
                for thresh in thresholds:
                    preds = [1 if s >= thresh else 0 for s in sent_scores]
                    tp = sum(1 for p, l in zip(preds, sent_labels) if p == 1 and l == 1)
                    fp = sum(1 for p, l in zip(preds, sent_labels) if p == 1 and l == 0)
                    fn = sum(1 for p, l in zip(preds, sent_labels) if p == 0 and l == 1)
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    f1_scores.append(f1)
                f1_tau = thresholds[np.argmax(f1_scores)]

                positive_rate = sum(1 for s in sent_scores if s >= f1_tau) / len(sent_scores)
                if positive_rate > 0.80:
                    print(f"  F1-optimal tau ({f1_tau:.4f}) flags {positive_rate:.0%} of sentences"
                          f" — falling back to 95th-percentile tau ({percentile_tau:.4f})")
                    self.threshold_ = percentile_tau
                else:
                    self.threshold_ = f1_tau
        else:
            self.threshold_ = percentile_tau

        print(f"  Anomaly threshold (tau): {self.threshold_:.4f}")
        return self

    def _mahalanobis(self, h_lex, h_syn):
        x = np.array([h_lex, h_syn]) - self.mu_
        return float(np.sqrt(x @ self.sigma_inv_ @ x))

    def score(self, texts):
        records = extract_features(texts, self.language_model)
        return [self._mahalanobis(r["h_lex"], r["h_syn"]) for r in records]

    def predict(self, texts):
        return [1 if s >= self.threshold_ else 0 for s in self.score(texts)]


def evaluate_full_system(normal_texts, phishing_texts):
    print("\n=== D3 Full System Evaluation (Joint H_lex + H_syn) ===")

    norm_trainval, norm_test = train_test_split(normal_texts, test_size=0.2, random_state=42)
    norm_train, norm_val = train_test_split(norm_trainval, test_size=0.125, random_state=42)
    _, phish_test = train_test_split(phishing_texts, test_size=0.2, random_state=42)
    _, phish_val = train_test_split(phishing_texts, test_size=0.125, random_state=42)

    print(f"  Train: {len(norm_train)} normal   Val: {len(norm_val)} normal + {len(phish_val)} phishing   Test: {len(norm_test)} normal + {len(phish_test)} phishing")

    val_texts = norm_val + phish_val
    val_labels = [0] * len(norm_val) + [1] * len(phish_val)

    detector = JointEntropyDetector()
    detector.fit(norm_train, val_texts=val_texts, val_labels=val_labels)

    norm_scores = detector.score(norm_test)
    phish_scores = detector.score(phish_test)

    all_scores = norm_scores + phish_scores
    all_labels = [0] * len(norm_scores) + [1] * len(phish_scores)
    preds = [1 if s >= detector.threshold_ else 0 for s in all_scores]
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
