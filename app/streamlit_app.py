"""
Entropy Dissonance — interactive research demo.
Run with: streamlit run app/streamlit_app.py
"""

import math
import json
import random
import sys
from collections import Counter
from pathlib import Path

# set non-interactive backend before anything that may import matplotlib
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                          # project root → src.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # app/ → plots

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import spacy

import src.adversarial as _adv_module  # triggers NLTK check + Agg confirm
from src.features import (
    NLP, sent_tokenize, word_tokenize,
    compute_h_lex, compute_h_syn,
    extract_features, NGRAM_ORDER,
)
from src.baseline import BaselineDetector
from src.full_system import JointEntropyDetector
from src.d1_baseline import D1SynDetector

import plots  # app/plots.py

DATA_DIR = ROOT / "data" / "processed"


# ─── Cached resources ──────────────────────────────────────────────────────

@st.cache_data
def _load_data():
    with open(DATA_DIR / "ham.json", encoding="utf-8") as f:
        ham = json.load(f)
    with open(DATA_DIR / "phishing.json", encoding="utf-8") as f:
        phish = json.load(f)
    return ham, phish


RESULTS_CSV = ROOT / "results" / "three_way_comparison.csv"


@st.cache_data
def load_results_csv():
    if not RESULTS_CSV.exists():
        return None
    df = pd.read_csv(RESULTS_CSV)
    required_cols = {"detector", "features", "auroc_clean", "auroc_adversarial", "delta"}
    if not required_cols.issubset(df.columns):
        return None
    if not {"D1", "D2", "D3"}.issubset(set(df["detector"])):
        return None
    return df.set_index("detector")


@st.cache_resource(show_spinner="Fitting detectors — first load takes ~3 min, cached after.")
def _init_all():
    ham, phish = _load_data()

    # splits mirror evaluate_baseline / evaluate_full_system exactly
    norm_trainval, norm_test  = train_test_split(ham,          test_size=0.2,   random_state=42)
    norm_train_d3, norm_val   = train_test_split(norm_trainval, test_size=0.125, random_state=42)
    _,             phish_test = train_test_split(phish,         test_size=0.2,   random_state=42)
    _,             phish_val  = train_test_split(phish,         test_size=0.125, random_state=42)

    val_texts  = norm_val  + phish_val
    val_labels = [0] * len(norm_val) + [1] * len(phish_val)

    # D1: H_syn only — same split as D2 (norm_trainval = 80 %)
    d1_det = D1SynDetector()
    d1_det.fit(norm_trainval)

    # D2: H_lex baseline
    d2_det = BaselineDetector()
    d2_det.fit(norm_trainval)

    # D3: joint Mahalanobis
    d3_det = JointEntropyDetector()
    d3_det.fit(norm_train_d3, val_texts=val_texts, val_labels=val_labels)

    # feature arrays for scatter plots — 500-email subsample is sufficient for viz
    viz_recs   = extract_features(norm_train_d3[:500], d3_det.language_model)
    phish_recs = extract_features(phish_test,           d3_det.language_model)

    norm_hl  = np.array([r["h_lex"] for r in viz_recs])
    norm_hs  = np.array([r["h_syn"] for r in viz_recs])
    phish_hl = np.array([r["h_lex"] for r in phish_recs])
    phish_hs = np.array([r["h_syn"] for r in phish_recs])

    # pre-scored test sentences needed for adversarial AUROC on page 6
    d1_norm_scores = d1_det.score(norm_test)
    d2_norm_scores = d2_det.score(norm_test)
    d3_norm_scores = d3_det.score(norm_test)

    return dict(
        d1_det=d1_det, d2_det=d2_det, d3_det=d3_det,
        norm_test=norm_test, phish_test=phish_test,
        ham=ham, phish=phish,
        norm_hl=norm_hl, norm_hs=norm_hs,
        phish_hl=phish_hl, phish_hs=phish_hs,
        d1_norm_scores=d1_norm_scores,
        d2_norm_scores=d2_norm_scores,
        d3_norm_scores=d3_norm_scores,
    )


def _run_adversarial(rate_pct, res):
    """
    Obfuscate all 259 phish_test emails at rate_pct %.
    Monkey-patches src.adversarial.SUBSTITUTION_RATE temporarily;
    safe because Streamlit processes one session at a time.
    """
    rate = rate_pct / 100.0
    old_rate = _adv_module.SUBSTITUTION_RATE
    _adv_module.SUBSTITUTION_RATE = rate
    try:
        rng = random.Random(_adv_module.RANDOM_SEED)
        obf_phish = [_adv_module._obfuscate_text(t, rng) for t in res["phish_test"]]
    finally:
        _adv_module.SUBSTITUTION_RATE = old_rate

    lm = res["d3_det"].language_model
    orig_recs = extract_features(res["phish_test"], lm)
    obf_recs  = extract_features(obf_phish, lm)

    def _auroc(det, norm_scores, obf_texts):
        obf_scores = det.score(obf_texts)
        labels = [0] * len(norm_scores) + [1] * len(obf_scores)
        return roc_auc_score(labels, norm_scores + obf_scores)

    return dict(
        orig_hl=[r["h_lex"] for r in orig_recs],
        obf_hl =[r["h_lex"] for r in obf_recs],
        orig_hs=[r["h_syn"] for r in orig_recs],
        obf_hs =[r["h_syn"] for r in obf_recs],
        d1_auroc=_auroc(res["d1_det"], res["d1_norm_scores"], obf_phish),
        d2_auroc=_auroc(res["d2_det"], res["d2_norm_scores"], obf_phish),
        d3_auroc=_auroc(res["d3_det"], res["d3_norm_scores"], obf_phish),
    )


# ─── Page functions ────────────────────────────────────────────────────────

def p1_what_features_see(res):
    st.header("What the features see")
    st.caption(
        "Edit the sentence below, then press **Analyse**. "
        "See how each token contributes to H_lex and how syntactic structure shapes H_syn."
    )

    DEFAULT = (
        "Urgent: your account has been suspended, click here to verify "
        "immediately or your access will be permanently revoked."
    )
    sentence = st.text_area("Input sentence", value=DEFAULT, height=80, key="p1_sent")
    analyse = st.button("Analyse", key="p1_btn")

    if not analyse and sentence == DEFAULT:
        # auto-render default on first load
        analyse = True

    if analyse:
        s = sentence.strip()
        if not s:
            st.info("Enter a sentence above.")
            return

        lm = res["d2_det"].language_model
        tokens = [t.lower() for t in word_tokenize(s) if t.isalpha()]

        if len(tokens) < 2:
            st.warning("Need at least 2 alphabetic tokens for H_lex.")
            return

        # (a) per-token surprisal table
        rows = []
        for i in range(1, len(tokens)):
            ctx  = tuple(tokens[max(0, i - NGRAM_ORDER + 1):i])
            prob = lm.score(tokens[i], ctx)
            surp = -math.log2(prob) if prob > 0 else None
            rows.append({
                "token":           tokens[i],
                "context":         " → ".join(ctx) or "(start)",
                "log₂P":           round(math.log2(prob), 4) if prob > 0 else "−∞",
                "surprisal (bits)": round(surp, 3) if surp else "∞",
            })

        st.subheader("(a) Per-token surprisal")
        st.dataframe(rows, use_container_width=True)

        valid = [(r["token"], r["surprisal (bits)"])
                 for r in rows if isinstance(r["surprisal (bits)"], float)]
        if valid:
            toks, surps = zip(*valid)
            st.plotly_chart(plots.surprisal_bar(list(toks), list(surps)), use_container_width=True)

        # (b) dependency parse via displaCy
        st.subheader("(b) Dependency parse")
        doc      = NLP(s)
        dep_html = spacy.displacy.render(doc, style="dep", page=True,
                                          options={"compact": True, "distance": 90})
        components.html(dep_html, height=340, scrolling=True)

        # (c) arc-label bar chart
        label_counts = Counter(tok.dep_ for tok in doc if tok.dep_)
        st.subheader("(c) Dependency-label distribution")
        st.plotly_chart(plots.dep_label_bar(label_counts), use_container_width=True)

        # (d) final values
        h_lex = compute_h_lex(s, lm)
        h_syn = compute_h_syn(s)
        st.subheader("(d) Feature values")
        c1, c2 = st.columns(2)
        c1.metric("H_lex  (mean surprisal)", f"{h_lex:.3f} bits" if h_lex is not None else "N/A")
        c2.metric("H_syn  (dep-label entropy)", f"{h_syn:.3f} bits" if h_syn is not None else "N/A")


def p2_feature_space(res):
    st.header("The feature space")
    st.caption(
        "Each point is one sentence from the corpus. "
        "Normal mean H_lex = 12.70, phishing mean H_lex = 14.45 (Δ = 1.76 bits, ~1.4σ). "
        "H_syn means differ by only 0.03 bits — the y-axis carries little class information."
    )

    c1, c2   = st.columns(2)
    show_n   = c1.checkbox("Normal training points",  value=True,  key="p2_n")
    show_p   = c2.checkbox("Phishing test points",    value=False, key="p2_p")
    show_kn  = c1.checkbox("KDE contours — normal",   value=False, key="p2_kn")
    show_kp  = c2.checkbox("KDE contours — phishing", value=False, key="p2_kp")

    fig = plots.scatter_feature_space(
        res["norm_hl"], res["norm_hs"],
        res["phish_hl"], res["phish_hs"],
        show_normal=show_n,
        show_phishing=show_p,
        show_kde_normal=show_kn,
        show_kde_phishing=show_kp,
    )
    st.plotly_chart(fig, use_container_width=True)


def p3_three_detectors(res):
    st.header("The three detectors")
    st.caption(
        "D3's Mahalanobis ellipse must extend along the H_syn axis where phishing overlaps normal — "
        "that geometric compromise is why its AUROC is lower than D2's single vertical threshold."
    )

    csv = load_results_csv()
    if csv is None:
        st.error(
            "Cannot display AUROC metrics: `results/three_way_comparison.csv` is missing or "
            "malformed. Run `python src/d1_baseline.py` to regenerate it."
        )
        st.stop()
    d1_clean = float(csv.loc["D1", "auroc_clean"])
    d2_clean = float(csv.loc["D2", "auroc_clean"])
    d3_clean = float(csv.loc["D3", "auroc_clean"])
    d3_vs_d2 = d3_clean - d2_clean
    m1, m2, m3 = st.columns(3)
    m1.metric("D1  H_syn only",   f"{d1_clean:.4f}", delta="below chance")
    m2.metric("D2  H_lex only",    f"{d2_clean:.4f}")
    m3.metric("D3  H_lex + H_syn", f"{d3_clean:.4f}", delta=f"{d3_vs_d2:+.4f} vs D2")
    st.caption(
        "Source: `results/three_way_comparison.csv` (produced by `src/d1_baseline.py`). "
        "Re-run that script to refresh."
    )

    show_det = st.radio(
        "Boundary to show",
        ["All", "D1", "D2", "D3", "None"],
        horizontal=True,
        key="p3_det",
    )
    show_p = st.checkbox("Show phishing test points", value=True, key="p3_p")

    if show_det == "None":
        fig = plots.scatter_feature_space(
            res["norm_hl"], res["norm_hs"],
            res["phish_hl"], res["phish_hs"],
            show_normal=True, show_phishing=show_p,
        )
    else:
        fig = plots.scatter_with_boundaries(
            res["norm_hl"], res["norm_hs"],
            res["phish_hl"], res["phish_hs"],
            d1_det=res["d1_det"],
            d2_det=res["d2_det"],
            d3_det=res["d3_det"],
            show_detectors=show_det,
            show_phishing=show_p,
        )
    st.plotly_chart(fig, use_container_width=True)


def p4_score_email(res):
    st.header("Score an email")
    st.caption(
        "Enter any email body. The table shows per-sentence anomaly scores for all three detectors. "
        "The evaluation is sentence-level — one anomalous sentence is enough to flag the email."
    )

    if "p4_email" not in st.session_state:
        st.session_state.p4_email = ""

    c1, c2 = st.columns(2)
    if c1.button("Load normal example"):
        st.session_state.p4_email = res["norm_test"][0]
    if c2.button("Load phishing example"):
        st.session_state.p4_email = res["phish_test"][0]

    email_text = st.text_area("Email body", value=st.session_state.p4_email,
                               height=200, key="p4_textarea")

    if not st.button("Score", key="p4_score"):
        return

    if not email_text.strip():
        st.warning("Paste an email above first.")
        return

    d1 = res["d1_det"]
    d2 = res["d2_det"]
    d3 = res["d3_det"]
    lm = d2.language_model  # D2's LM for H_lex

    sents = sent_tokenize(email_text)
    rows   = []
    ep_hl  = []
    ep_hs  = []

    for s in sents:
        h_lex = compute_h_lex(s, lm)
        h_syn = compute_h_syn(s)

        d1_score = (((h_syn - d1.mean_) / d1.std_) ** 2) if h_syn is not None else None
        d2_score = h_lex
        d3_score = (d3._mahalanobis(h_lex, h_syn)
                    if h_lex is not None and h_syn is not None else None)

        flagged = []
        if d1_score is not None and d1_score > d1.threshold_:
            flagged.append("D1")
        if d2_score is not None and d2_score > d2.threshold_:
            flagged.append("D2")
        if d3_score is not None and d3_score > d3.threshold_:
            flagged.append("D3")

        rows.append({
            "Sentence":   s[:90] + ("…" if len(s) > 90 else ""),
            "H_lex":      f"{h_lex:.3f}"   if h_lex  is not None else "—",
            "H_syn":      f"{h_syn:.3f}"   if h_syn  is not None else "—",
            "D1 score":   f"{d1_score:.3f}" if d1_score is not None else "—",
            "D2 score":   f"{d2_score:.3f}" if d2_score is not None else "—",
            "D3 score":   f"{d3_score:.3f}" if d3_score is not None else "—",
            "Flagged by": ", ".join(flagged) if flagged else "—",
        })
        if h_lex is not None and h_syn is not None:
            ep_hl.append(h_lex)
            ep_hs.append(h_syn)

    st.subheader("Per-sentence scores")
    st.dataframe(rows, use_container_width=True)

    # email-level aggregate
    d1_vals = [float(r["D1 score"]) for r in rows if r["D1 score"] != "—"]
    d2_vals = [float(r["D2 score"]) for r in rows if r["D2 score"] != "—"]
    d3_vals = [float(r["D3 score"]) for r in rows if r["D3 score"] != "—"]

    st.subheader("Email-level aggregate (max sentence score vs threshold)")
    a1, a2, a3 = st.columns(3)
    if d1_vals:
        mx = max(d1_vals)
        a1.metric("D1 max", f"{mx:.3f}",
                  delta=f"thresh {d1.threshold_:.1f} {'⚠ FLAGGED' if mx > d1.threshold_ else ''}",
                  delta_color="inverse" if mx > d1.threshold_ else "off")
    if d2_vals:
        mx = max(d2_vals)
        a2.metric("D2 max H_lex", f"{mx:.3f}",
                  delta=f"thresh {d2.threshold_:.2f} {'⚠ FLAGGED' if mx > d2.threshold_ else ''}",
                  delta_color="inverse" if mx > d2.threshold_ else "off")
    if d3_vals:
        mx = max(d3_vals)
        a3.metric("D3 max Mahal.", f"{mx:.3f}",
                  delta=f"thresh {d3.threshold_:.2f} {'⚠ FLAGGED' if mx > d3.threshold_ else ''}",
                  delta_color="inverse" if mx > d3.threshold_ else "off")

    if ep_hl:
        st.subheader("Sentence positions in the feature space")
        fig = plots.scatter_feature_space(
            res["norm_hl"], res["norm_hs"],
            res["phish_hl"], res["phish_hs"],
            show_normal=True, show_phishing=True,
            extra_points=[{
                "hl": ep_hl, "hs": ep_hs,
                "label": "This email (sentences)",
                "color": "#d62728",
            }],
        )
        st.plotly_chart(fig, use_container_width=True)


def p5_correlation(res):
    st.header("Correlation structure")
    st.caption(
        "The dependence between lexical surprisal and syntactic entropy differs sharply between classes, "
        "even though H_syn's marginal alone does not separate them. "
        "This structural difference suggests correlation-aware scoring as a future direction; "
        "the current Mahalanobis approach uses the normal-data covariance and so is partially "
        "sensitive to this difference but does not directly exploit it."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = plots.correlation_scatter(
            res["norm_hl"], res["norm_hs"],
            title="Normal training (~5 k sentences)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = plots.correlation_scatter(
            res["phish_hl"], res["phish_hs"],
            title="Phishing test (~1.5 k sentences)",
        )
        st.plotly_chart(fig, use_container_width=True)


def p6_adversarial(res):
    st.header("Adversarial: the OOV artefact")
    st.caption(
        "Rising AUROC under synonym attack is an artefact, not robustness. "
        "Substituted tokens form unseen trigrams → Laplace LM assigns high surprisal → H_lex inflates. "
        "H_syn (right plot) is unchanged because syntactic structure is preserved by same-POS substitution. "
        "D1's near-flat AUROC across all rates confirms the artefact lives entirely in the H_lex component."
    )

    rate = st.slider("Substitution rate (%)", 0, 50, 30, step=5, key="p6_rate")

    if st.button("Run adversarial evaluation at this rate", key="p6_run"):
        if "p6_cache" not in st.session_state:
            st.session_state.p6_cache = {}
        if rate not in st.session_state.p6_cache:
            with st.spinner(f"Obfuscating 259 phishing emails at {rate}% — may take 2–3 min…"):
                result = _run_adversarial(rate, res)
            st.session_state.p6_cache[rate] = result

            # reproducibility check at 30 %
            if rate == 30:
                csv = load_results_csv()
                if csv is not None:
                    expected = {
                        "D1": float(csv.loc["D1", "auroc_adversarial"]),
                        "D2": float(csv.loc["D2", "auroc_adversarial"]),
                        "D3": float(csv.loc["D3", "auroc_adversarial"]),
                    }
                    mismatch = []
                    for k, exp in expected.items():
                        actual = result[f"{k.lower()}_auroc"]
                        if abs(actual - exp) > 0.0005:
                            mismatch.append(f"{k}: got {actual:.4f}, expected {exp:.4f}")
                    if mismatch:
                        st.error(
                            "REPRODUCIBILITY MISMATCH at 30%. "
                            "The obfuscation pipeline may differ from adversarial.py:\n"
                            + "\n".join(mismatch)
                        )
                    else:
                        st.success("✓ Adversarial AUROCs at 30% match expected values.")
        else:
            st.info(f"Cached result for {rate}%.")

    if "p6_cache" in st.session_state and rate in st.session_state.p6_cache:
        r = st.session_state.p6_cache[rate]

        csv = load_results_csv()
        d1_ref = f"clean {float(csv.loc['D1', 'auroc_clean']):.4f}" if csv is not None else None
        d2_ref = f"clean {float(csv.loc['D2', 'auroc_clean']):.4f}" if csv is not None else None
        d3_ref = f"clean {float(csv.loc['D3', 'auroc_clean']):.4f}" if csv is not None else None
        m1, m2, m3 = st.columns(3)
        m1.metric("D1 AUROC", f"{r['d1_auroc']:.4f}", delta=d1_ref, delta_color="off")
        m2.metric("D2 AUROC", f"{r['d2_auroc']:.4f}", delta=d2_ref, delta_color="off")
        m3.metric("D3 AUROC", f"{r['d3_auroc']:.4f}", delta=d3_ref, delta_color="off")

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**H_lex distribution** — shifts right under attack")
            fig = plots.h_distribution(r["orig_hl"], r["obf_hl"], "H_lex")
            st.plotly_chart(fig, use_container_width=True)
        with fc2:
            st.markdown("**H_syn distribution** — should NOT shift")
            fig = plots.h_distribution(r["orig_hs"], r["obf_hs"], "H_syn")
            st.plotly_chart(fig, use_container_width=True)


# ─── Main / sidebar ────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Entropy Dissonance",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Entropy Dissonance: A Diagnostic Study of Lexical-Syntactic Phishing Detection")

    with st.sidebar:
        st.markdown("### Navigate")
        page = st.radio(
            "Select page",
            [
                "1 · What the features see",
                "2 · The feature space",
                "3 · The three detectors",
                "4 · Score an email",
                "5 · Correlation structure",
                "6 · Adversarial: OOV artefact",
            ],
            label_visibility="collapsed",
            key="nav_page",
        )
        st.markdown("---")
        with st.expander("About this demo"):
            st.markdown(
                "**Feature pair:** H_lex (mean per-token surprisal under a "
                "trigram Laplace language model) and H_syn (Shannon entropy "
                "over spaCy dependency-arc labels), computed per sentence.\n\n"
                "**Structural finding:** D3 (joint Mahalanobis) underperforms "
                "D2 (H_lex only) because H_syn carries near-zero class signal — "
                "D1 (H_syn alone) achieves AUROC 0.47, effectively below chance. "
                "Adding a non-discriminative second axis dilutes the H_lex signal.\n\n"
                "**Methodological caution:** Both D2 and D3 *improve* under synonym "
                "substitution. This is an artefact: substituted tokens create unseen "
                "trigrams that spike H_lex, making phishing *more* anomalous, not less. "
                "Not a robustness demonstration."
            )

    res = _init_all()

    if page.startswith("1"):
        p1_what_features_see(res)
    elif page.startswith("2"):
        p2_feature_space(res)
    elif page.startswith("3"):
        p3_three_detectors(res)
    elif page.startswith("4"):
        p4_score_email(res)
    elif page.startswith("5"):
        p5_correlation(res)
    elif page.startswith("6"):
        p6_adversarial(res)


if __name__ == "__main__":
    main()
