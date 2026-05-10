import math
import re
import warnings
from collections import Counter

import spacy

warnings.filterwarnings("ignore")

try:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    raise OSError("spaCy model not found. Run: python -m spacy download en_core_web_sm")

NGRAM_ORDER = 3

_ABBREV_SET = frozenset({
    'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'ltd',
    'corp', 'fig', 'vol', 'ed', 'approx', 'est', 'dept', 'no', 'st', 'ave',
    'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
})

# . only splits before a capital; ? and ! always split
_SENT_SPLIT_RE = re.compile(r'(?<=[!?])\s+|(?<=[.])\s+(?=[A-Z])')


def sent_tokenize(text):
    text = text.strip()
    if not text:
        return []

    candidates = list(_SENT_SPLIT_RE.finditer(text))
    sentences = []
    start = 0

    for match in candidates:
        end = match.start()
        chunk = text[start:end]
        word_match = re.search(r'([A-Za-z]+)[.!?]$', chunk)
        if word_match:
            w = word_match.group(1).lower()
            if w in _ABBREV_SET or len(w) == 1:
                continue  # abbreviation, not a sentence boundary
        sentences.append(chunk.strip())
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


_WORD_RE = re.compile(r'[a-zA-Z]+')


def word_tokenize(text):
    return _WORD_RE.findall(text)


class LaplaceTrigramModel:
    def __init__(self):
        self._unigrams = Counter()
        self._bigrams = Counter()
        self._trigrams = Counter()
        self._vocab_size = 0

    def fit(self, sentences):
        for tokens in sentences:
            for i, tok in enumerate(tokens):
                self._unigrams[tok] += 1
                if i >= 1:
                    self._bigrams[(tokens[i - 1], tok)] += 1
                if i >= 2:
                    self._trigrams[(tokens[i - 2], tokens[i - 1], tok)] += 1
        self._vocab_size = len(self._unigrams)
        return self

    def score(self, word, context):
        # context is a tuple of 0, 1, or 2 preceding tokens
        # P(word|context) = (count + 1) / (context_count + V)
        V = self._vocab_size or 1
        if len(context) >= 2:
            c_w = self._trigrams.get((context[-2], context[-1], word), 0)
            c_ctx = self._bigrams.get((context[-2], context[-1]), 0)
        elif len(context) == 1:
            c_w = self._bigrams.get((context[0], word), 0)
            c_ctx = self._unigrams.get(context[0], 0)
        else:
            c_w = self._unigrams.get(word, 0)
            c_ctx = sum(self._unigrams.values())
        return (c_w + 1) / (c_ctx + V)


def build_language_model(texts):
    sentences = []
    for text in texts:
        for sent in sent_tokenize(text):
            tokens = [t.lower() for t in word_tokenize(sent) if t.isalpha()]
            if tokens:
                sentences.append(tokens)

    model = LaplaceTrigramModel()
    model.fit(sentences)
    return model


def compute_h_lex(sentence, model):
    tokens = [t.lower() for t in word_tokenize(sentence) if t.isalpha()]
    if len(tokens) < 2:
        return None

    surprisals = []
    for i in range(1, len(tokens)):
        context = tuple(tokens[max(0, i - NGRAM_ORDER + 1):i])
        prob = model.score(tokens[i], context)
        if prob > 0:
            surprisals.append(-math.log2(prob))

    if not surprisals:
        return None
    return sum(surprisals) / len(surprisals)


def compute_h_syn(sentence):
    doc = NLP(sentence)
    labels = [token.dep_ for token in doc if token.dep_ != ""]

    if len(labels) < 2:
        return None

    counts = Counter(labels)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def extract_features(texts, model):
    records = []
    for doc_text in texts:
        for sent in sent_tokenize(doc_text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            h_lex = compute_h_lex(sent, model)
            h_syn = compute_h_syn(sent)
            if h_lex is not None and h_syn is not None:
                records.append({"text": sent, "h_lex": h_lex, "h_syn": h_syn})
    return records
