# Defence Preparation: Entropy Dissonance Detector

---

## 1. What the project does — plain English

You built a system that tries to detect phishing emails by measuring how "off" they sound statistically, without looking for specific words or phrases. The idea is that phishing emails use unusual word combinations (weird for someone writing a normal work email) and might have different grammatical patterns. You trained a language model on legitimate emails only, so it learned what normal email looks like. When you feed it a phishing email, it notices the unusual word choices and assigns a high "suspiciousness" score. You also tried adding a second signal based on grammar structure, to see if combining both signals beat using word patterns alone. The headline finding is that the grammar signal added nothing — and you figured out exactly why.

---

## 2. The five things you must be able to explain

---

### H_lex — how surprising are the words?

**Short version (for a quick question):**
H_lex is the average surprisal of each word in a sentence, measured in bits, using a trigram language model trained on normal email. A high H_lex value means the sentence contains word combinations that rarely appeared in normal email — which is a signal the sentence might be phishing.

**Long version (if he wants detail):**
First, a trigram language model is trained on legitimate emails only. It counts every sequence of one, two, and three consecutive words in those emails. For any new sentence, it looks at each word in turn and asks: given the previous two words, how often did this particular word follow them in training? That probability is estimated as (count of that three-word sequence + 1) divided by (count of the two-word context + vocabulary size). The +1 is Laplace smoothing — explained below. Surprisal for that word is minus log base 2 of that probability. If the probability is high (common word in context), surprisal is low. If the probability is low (unusual combination), surprisal is high. H_lex is just the average of all those surprisal values for every word in the sentence. Phishing sentences tend to use unusual phrasing that was never in the normal training data, so they score higher.

---

### H_syn — how varied is the grammar?

**Short version:**
H_syn is the Shannon entropy of the dependency arc labels in a sentence — basically, how spread out the grammatical roles are. spaCy parses the sentence, labels each word's role (subject, object, determiner, etc.), and you compute entropy over how evenly those labels are distributed.

**Long version:**
spaCy's dependency parser assigns every word a label that describes its grammatical relationship to the rest of the sentence. Words like "the" get "det" (determiner), the subject of a verb gets "nsubj", the direct object gets "dobj", and so on. You take all those labels for a sentence and treat them as a categorical distribution — count how many times each label appears, divide by total labels to get proportions, then apply the Shannon entropy formula: negative sum of (p × log₂p) for each label type. A sentence that uses many different label types roughly equally has high entropy. A sentence dominated by one or two label types has low entropy. The original hypothesis was that phishing sentences would show a different pattern of grammatical roles than normal sentences. The result showed they do not — the H_syn distributions of normal and phishing sentences overlap almost completely.

---

### Laplace smoothing — why it matters

**Short version:**
Laplace smoothing adds 1 to every word count so that word sequences the model never saw in training still get a tiny non-zero probability instead of zero. This matters because a probability of zero would make the log undefined, and phishing emails are full of unusual word combinations the model never saw.

**Long version:**
Without smoothing, if a phishing sentence contains a word sequence that never appeared in the legitimate training emails, the count is zero, the probability is zero, and log₂(0) is undefined — the whole calculation breaks. Laplace smoothing fixes this by pretending you saw every possible word one extra time, even words and sequences you never actually saw. The formula becomes (count + 1) divided by (context count + vocabulary size). The vocabulary size in the denominator ensures probabilities still sum to 1 across all possible next words. The critical consequence for the adversarial experiment: when synonym substitution creates a new three-word sequence that was never in training, the count is 0, so the probability becomes 1 divided by vocabulary size. With a vocabulary of around 40,000 words, that probability is about 0.000025, and surprisal is about 15 bits. That is a huge spike — it makes the substituted sentence look extremely anomalous, which is why AUROC went up under attack instead of down.

---

### Mahalanobis distance — why not just use a threshold?

**Short version:**
Mahalanobis distance measures how far a point is from the centre of a data cloud, accounting for the shape and correlation of the cloud. It is better than separate thresholds because it treats the two features as a joint distribution and accounts for the fact that they are correlated and at different scales.

**Long version:**
D2 puts a single threshold on H_lex and flags anything above it. That works in one dimension. D3 has two features — H_lex and H_syn — and needs to decide whether a (H_lex, H_syn) point is anomalous. You cannot just put a threshold on each feature independently and AND them together, because H_lex and H_syn are correlated (r = 0.44 on the normal training set) and operate at different scales (H_lex is typically 11–15 bits, H_syn is typically 1.5–4 bits). Mahalanobis distance solves this by computing the 2×2 covariance matrix of the normal training data, inverting it, and using that inverse to scale and rotate the feature space. Geometrically: it transforms the elliptical normal data cloud into a circle and then measures how far a test point is from the centre. The formula is √((x − μ)ᵀ Σ⁻¹ (x − μ)) where μ is the mean of the normal training features and Σ is the covariance matrix. In this project the covariance matrix is computed manually using a triple nested loop in full_system.py, which is worth being able to explain — it subtracts the mean, accumulates outer products, and divides by n−1 for sample covariance.

---

### The adversarial experiment — what it did and what the result means

**Short version:**
The experiment replaced 30% of content words in phishing emails with WordNet synonyms, then re-evaluated the detectors. Both D2 and D3 got better at detecting the modified phishing emails — AUROC went up. This happened because the synonym substitutions created word sequences the language model had never seen, making the phishing emails appear even more anomalous than before.

**Long version:**
The original intention was to test whether an attacker using synonym substitution could make phishing emails "blend in" better with normal email and fool the detectors. You used NLTK's WordNet to find synonyms for nouns, verbs, and adjectives, and randomly replaced 30% of eligible words in each phishing sentence. The results were: D2 AUROC went from 0.8087 to 0.8561, and D3 AUROC went from 0.7136 to 0.7789. Both improved. The explanation is Laplace smoothing: the original phishing sentences used words that appeared in some ham context — maybe the word "verify" appeared once in a ham email about verifying a booking. After substitution, the replacement word (say "confirm") in that same context creates a brand new three-word sequence the model has never seen at all. That gets Laplace's minimum probability and a huge surprisal spike. So instead of blending in, the attacked sentences became more obviously anomalous. The D1 detector (H_syn only) barely changed under the attack — its AUROC went from 0.4677 to 0.4923 — which confirms the whole effect is in the H_lex component. Synonym substitution preserves grammatical structure (same part-of-speech replacements), so H_syn is unaffected.

---

## 3. Questions the lecturer will almost certainly ask — and exactly how to answer

---

**"Why does your full system D3 perform worse than the simpler D2 baseline?"**

The direct answer is that H_syn has almost no ability to distinguish phishing from normal email on its own. I can prove this with the D1 detector — H_syn only — which achieved AUROC of 0.4677, which is below random chance. The mean H_syn value differs by only 0.03 bits between normal and phishing sentences, which is less than 5% of one standard deviation. When you add a feature to Mahalanobis distance that carries no discriminative signal, you are adding a second axis that the decision ellipse has to accommodate without getting any useful information from it. The ellipse has to stretch in the H_syn direction to fit the normal training data shape, and that stretching dilutes the H_lex signal. The covariance matrix is well-conditioned (condition number 5.5), so numerical instability is not the cause. It is simply that H_syn is not a useful feature for this problem.

---

**"What does D1 AUROC of 0.4677 mean? Is that a bug?"**

No. AUROC below 0.5 means the detector is scoring phishing sentences lower than normal sentences — it is working in the wrong direction. For D1, high anomaly score means far from the normal H_syn mean. But phishing H_syn values are actually concentrated near the normal mean, not in the tails. Many normal sentences — short ones, informal ones — have unusual H_syn values. So the normal distribution has heavier tails than phishing, and those normal outliers score higher than average phishing sentences. The result is that if you flip the D1 score (flag low scores instead of high scores), you would get AUROC of 1 − 0.4677 = 0.5323, which is barely better than random. Either way, H_syn simply does not separate the classes. An AUROC of 0.4677 is not a programming error — it is an empirical finding that the feature does not work.

---

**"Your AUROC goes up after the adversarial attack. How is that not a problem with your evaluation?"**

It is a problem with the framing, not the evaluation. The evaluation is measuring exactly what it should measure — whether the detector can distinguish obfuscated phishing from normal email. The answer is yes, and it does so more successfully after obfuscation. The problem is with the claim the experiment was supposed to test. I originally framed it as a robustness test — "can an attacker use synonym substitution to evade detection?" The result shows they cannot. In fact, synonym substitution makes phishing more detectable, because the language model was trained on real ham email, and WordNet synonyms in that context produce word sequences the model never saw. That creates huge surprisal spikes via Laplace smoothing. The D1 baseline (H_syn only) barely changes under attack, confirming the entire effect is in the H_lex component — exactly what you would expect if the mechanism is OOV trigram inflation.

---

**"Why trigram and not bigram or 4-gram?"**

Trigrams capture two-word context for each prediction, which is the standard balance in n-gram language modelling between context richness and data sparsity. Bigrams give only one word of context, which is often too little to distinguish phishing phrasing from normal phrasing — the word "click" is common in both, but "click here" and "click here to" are more distinctive. 4-grams require more training data to get reliable counts because there are exponentially more possible 4-gram sequences. With a training set of around 3,000 emails, higher-order models would be severely sparse. Trigrams are a principled choice for this data scale. That said, this was not empirically validated — it would be a fair criticism to say I should have compared bigram, trigram, and 4-gram performance.

---

**"Why Laplace smoothing and not Kneser-Ney?"**

Kneser-Ney is the standard for production n-gram language models because it distributes back-off probability based on how often a word appears in diverse contexts, giving better calibrated probability estimates. I used Laplace because it is much simpler to implement from scratch without relying on libraries. In hindsight, the choice of smoothing method actually matters critically for the adversarial experiment result — Laplace gives the same low probability to every unseen trigram regardless of how common the individual words are, which exaggerates the surprisal spike. Kneser-Ney would assign higher back-off probability to common words, potentially reducing the OOV inflation effect. Whether that would improve or hurt AUROC is an open question. This is a legitimate limitation of the approach.

---

**"Why fit on normal email only? Why not use phishing data in training?"**

This is one-class anomaly detection. The idea is that you may have very little phishing data or you may encounter phishing tactics you have never seen before. If you train a two-class classifier (normal vs phishing), it can only detect the phishing patterns it was trained on. A one-class model trained on normal data can flag anything that looks unlike normal email, including novel attacks. This is a design choice common in anomaly detection for security applications. The limitation is that if phishing emails use language that closely resembles normal email (which is increasingly true with modern AI-generated phishing), one-class models struggle.

---

**"Why Mahalanobis and not just Euclidean distance?"**

Euclidean distance treats both features as equally important and independent, and measures straight-line distance. If H_lex has standard deviation 1.5 bits and H_syn has standard deviation 0.6 bits, a point that is 2 bits away in H_lex and 2 bits away in H_syn is not equally anomalous in both directions — the H_syn deviation is more extreme relative to the spread of normal data. Mahalanobis distance accounts for this by normalising by the covariance structure. It also accounts for the correlation between the features — if H_lex and H_syn tend to move together in normal email, a point that deviates in both directions simultaneously is more anomalous than the two individual deviations suggest. Geometrically, Euclidean distance treats the normal data cloud as a sphere; Mahalanobis distance treats it as the ellipsoid it actually is.

---

**"What is your contribution in one sentence?"**

This project demonstrates that sentence-level lexical surprisal under a trigram language model is an effective signal for phishing detection (AUROC 0.81), that dependency-label entropy adds no discriminative value to that signal (D1 AUROC 0.47, below chance), and that synonym substitution attacks on the lexical signal actually increase detectability rather than reducing it — an artefact of Laplace-smoothed trigram surprisal on out-of-vocabulary sequences.

---

**"What are the limitations?"**

Four honest ones. First, evaluation is at sentence level — a phishing email with mostly normal sentences and one suspicious one might score differently depending on whether you take the max sentence score or the mean. Second, the adversarial experiment is not a genuine evasion demonstration — the attack backfired, and a smarter attacker who understood the language model could in principle craft sequences that look normal. Third, Laplace smoothing is a poor choice for large vocabularies — Kneser-Ney would produce better-calibrated surprisal estimates. Fourth, there is no paper accompanying this code, which is half the rubric.

---

## 4. Code you must be able to point to and explain

**`src/features.py` — `LaplaceTrigramModel.score()` (lines 78–91)**
This is the core of H_lex. The `if len(context) >= 2` branch is the trigram formula: count the three-word sequence, count the two-word context, apply smoothing. The `elif len(context) == 1` branch falls back to bigrams for the second word of a sentence. The `else` branch uses unigrams for truly context-free scoring. The return line `(c_w + 1) / (c_ctx + V)` is the complete Laplace formula. Be able to trace through an example: if context is ("please", "verify") and word is "your", and that trigram has count 0 but ("please", "verify") has count 3, and V = 40000, the probability is 1 / (3 + 40000) ≈ 0.000025.

**`src/features.py` — `compute_h_lex()` (lines 107–121)**
The loop starts at `i=1`, not `i=0`, because the first token has no preceding context. For each token, it builds a context tuple of up to the previous two tokens (line 114: `max(0, i - NGRAM_ORDER + 1)`), gets the probability from the model, computes `-log₂(prob)`, and appends to the surprisals list. The return is `sum(surprisals) / len(surprisals)` — a mean, not a sum. Note: H_lex is not entropy, it is mean surprisal.

**`src/features.py` — `compute_h_syn()` (lines 124–138)**
Runs the full spaCy pipeline on the sentence. Collects all non-empty dependency labels. Counts them, computes proportions, applies Shannon entropy formula. Returns `None` if fewer than 2 labels (too short to be meaningful).

**`src/full_system.py` — `JointEntropyDetector.fit()` (lines 14–79)**
Lines 22–36 fit the bivariate Gaussian: compute mean (line 22), then compute sample covariance with a triple nested loop (lines 24–31), then invert with `np.linalg.inv` (line 34), falling back to `pinv` if singular (line 36). Lines 44–74 set the threshold — first tries to find the F1-optimal threshold on a validation set, falls back to the 95th percentile of normal training distances if the optimal threshold would flag more than 80% of sentences.

**`src/full_system.py` — `_mahalanobis()` (lines 81–83)**
One line: `float(np.sqrt(x @ self.sigma_inv_ @ x))` where `x = np.array([h_lex, h_syn]) - self.mu_`. This is exactly √((x − μ)ᵀ Σ⁻¹ (x − μ)). The `@` operator is matrix multiplication.

**`src/adversarial.py` — `_obfuscate_sentence()` (lines 51–66)**
Tokenises the sentence with NLTK's `word_tokenize` (different tokeniser from `features.py` — this is a real inconsistency in the code). POS-tags each token. Finds eligible tokens (nouns, verbs, adjectives with WordNet entries). Randomly samples 30% of them and replaces each with the first WordNet synonym that is not the original word. Returns the reconstructed sentence as a joined string.

---

## 5. Things you must NOT say

**Do not say H_lex is "entropy."**
H_lex is mean surprisal. Entropy is the expected surprisal over a distribution, computed with a weighted sum. H_lex is a simple average over the tokens in one sentence. They are related concepts but they are not the same thing. If you call H_lex entropy, an examiner will correct you immediately and it will signal you do not understand your own feature.

**Do not say D3 failed because the covariance matrix was singular.**
It is well-conditioned. Condition number is approximately 5.5 and determinant is approximately 0.45. Singular matrices have condition numbers approaching infinity and determinant of zero. The `pinv` fallback in the code is a defensive precaution, not evidence that singularity occurred. The real reason D3 underperforms is that H_syn has no discriminative signal.

**Do not say the adversarial attack "worked" or that your system is "robust."**
Neither is true. AUROC went up, not down. The attack backfired. If you say your system is robust to synonym substitution, you will be asked to explain why AUROC increased, and you will not have a good answer. The correct framing is: synonym substitution using this method does not evade detection; it increases detectability as a side effect of Laplace smoothing on unseen trigrams. That is an interesting empirical finding, not evidence of robustness.

**Do not say the adversarial experiment printed "consistent with the stated hypothesis."**
Look at `adversarial.py` lines 151–157. That print statement fires only when D2 degraded more than D3. The actual printed output showed both improved, so the `else` branch fired: "D3 degraded as much or more than D2 under synonym substitution. This does not confirm the hypothesis." Know what your code actually printed.

**Do not describe this as a machine learning classifier.**
There is no training in the supervised sense. D2 and D1 are anomaly detectors that fit a Gaussian on normal data only. D3 is a one-class Mahalanobis detector. You are not training a model to distinguish two classes — you are modelling one class (normal) and flagging deviations. If you describe it as a classifier that "learned to tell phishing from ham," you are misrepresenting the architecture.

**Do not say the 20-character sentence filter removes short sentences.**
The filter is on character length, not token length. The sentence "Click here now." is 15 characters and would be filtered, but it has 3 tokens. The sentence "I'd like a receipt." is 20 characters and might pass, but it has 5 tokens. This is actually a weak design choice — a token-length filter would be more principled for controlling entropy estimator reliability. If asked about it, acknowledge that a character filter is a rough proxy and a token-count filter would be more appropriate.

**Do not claim the correlation finding was the original hypothesis.**
The correlation difference between classes (r = 0.44 normal, r = 0.08 phishing) was discovered during diagnostic analysis after the main results were in. It was not predicted in advance. It is an interesting structural finding, but presenting it as the planned contribution would be inaccurate.

---

## Quick reference: numbers to know cold

| What | Value |
|------|-------|
| D1 AUROC (H_syn only) | 0.4677 |
| D2 AUROC (H_lex only) | 0.8087 |
| D3 AUROC (joint) | 0.7136 |
| D2 AUROC after attack | 0.8561 |
| D3 AUROC after attack | 0.7789 |
| D1 AUROC after attack | 0.4923 |
| H_lex mean difference (normal vs phishing) | ~1.76 bits |
| H_syn mean difference (normal vs phishing) | ~0.03 bits |
| Normal training emails | 3,096 (80% of 3,871 ham) |
| Substitution rate in adversarial experiment | 30% |
| Pearson r, H_lex vs H_syn (normal) | 0.44 |
| Pearson r, H_lex vs H_syn (phishing) | 0.08 |
| Covariance matrix condition number | ~5.5 |
| Laplace formula | (count + 1) / (context_count + V) |
| Mahalanobis formula | √((x − μ)ᵀ Σ⁻¹ (x − μ)) |
| Shannon entropy formula | −Σ p_k × log₂(p_k) |
