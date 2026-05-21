"""
Pure plotting functions — no Streamlit imports here.
All functions return plotly Figure objects (or None on error).
"""

import math
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, pearsonr, linregress

# Base font size applied to every chart for projector visibility.
_FONT = dict(size=15)


# ─── Internal helpers ──────────────────────────────────────────────────────

def _kde_contour(x, y, color, name, n_grid=60):
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except (np.linalg.LinAlgError, ValueError):
        return None
    xi = np.linspace(float(np.min(x)) - 0.5, float(np.max(x)) + 0.5, n_grid)
    yi = np.linspace(float(np.min(y)) - 0.5, float(np.max(y)) + 0.5, n_grid)
    XX, YY = np.meshgrid(xi, yi)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    return go.Contour(
        x=xi, y=yi, z=ZZ,
        showscale=False,
        contours_coloring="lines",
        line_color=color,
        name=name,
        showlegend=True,
        opacity=0.8,
        ncontours=5,
        hoverinfo="skip",
    )


def _ellipse_points(mu, sigma_inv, tau, n=300):
    """Parametric (x, y) for the Mahalanobis = tau contour."""
    cov = np.linalg.inv(sigma_inv)
    vals, vecs = np.linalg.eigh(cov)
    theta = np.linspace(0, 2 * np.pi, n)
    unit = np.array([np.cos(theta), np.sin(theta)])
    ell = vecs @ np.diag(tau * np.sqrt(vals)) @ unit
    return mu[0] + ell[0], mu[1] + ell[1]


# ─── Public API ────────────────────────────────────────────────────────────

def scatter_feature_space(
    norm_hl, norm_hs, phish_hl, phish_hs,
    show_normal=True,
    show_phishing=False,
    show_kde_normal=False,
    show_kde_phishing=False,
    extra_points=None,
):
    """
    Base scatter: H_lex vs H_syn.

    extra_points: list of dicts with keys hl, hs, label, color.
    """
    fig = go.Figure()

    if show_normal:
        fig.add_trace(go.Scatter(
            x=list(norm_hl), y=list(norm_hs),
            mode="markers",
            marker=dict(color="#7b9ecc", opacity=0.12, size=4),
            name="Normal (train)",
            hovertemplate="H_lex=%{x:.3f}<br>H_syn=%{y:.3f}<extra>Normal</extra>",
        ))
    if show_kde_normal and show_normal and len(norm_hl) > 10:
        t = _kde_contour(norm_hl, norm_hs, "#3a6ea5", "KDE Normal")
        if t:
            fig.add_trace(t)

    if show_phishing:
        fig.add_trace(go.Scatter(
            x=list(phish_hl), y=list(phish_hs),
            mode="markers",
            marker=dict(color="#e07b54", opacity=0.30, size=5),
            name="Phishing (test)",
            hovertemplate="H_lex=%{x:.3f}<br>H_syn=%{y:.3f}<extra>Phishing</extra>",
        ))
    if show_kde_phishing and show_phishing and len(phish_hl) > 10:
        t = _kde_contour(phish_hl, phish_hs, "#a84e2f", "KDE Phishing")
        if t:
            fig.add_trace(t)

    if extra_points:
        for ep in extra_points:
            fig.add_trace(go.Scatter(
                x=ep["hl"], y=ep["hs"],
                mode="markers",
                marker=dict(color=ep.get("color", "#333"), size=10,
                            symbol="star", line=dict(width=1, color="black")),
                name=ep["label"],
                hovertemplate=(
                    "H_lex=%{x:.3f}<br>H_syn=%{y:.3f}"
                    f"<extra>{ep['label']}</extra>"
                ),
            ))

    fig.update_layout(
        xaxis_title="H_lex — mean per-token surprisal (bits)",
        yaxis_title="H_syn — dep-label entropy (bits)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=_FONT,
        height=500,
        margin=dict(l=60, r=20, t=40, b=60),
    )
    return fig


def scatter_with_boundaries(
    norm_hl, norm_hs, phish_hl, phish_hs,
    d1_det, d2_det, d3_det,
    show_detectors="All",
    show_phishing=True,
):
    """Scatter with D1 / D2 / D3 decision boundaries overlaid."""
    fig = scatter_feature_space(
        norm_hl, norm_hs, phish_hl, phish_hs,
        show_normal=True,
        show_phishing=show_phishing,
    )

    all_hl = list(norm_hl) + list(phish_hl)
    all_hs = list(norm_hs) + list(phish_hs)
    x_lo, x_hi = min(all_hl) - 0.5, max(all_hl) + 0.5
    y_lo, y_hi = min(all_hs) - 0.5, max(all_hs) + 0.5

    show_d1 = show_detectors in ("D1", "All")
    show_d2 = show_detectors in ("D2", "All")
    show_d3 = show_detectors in ("D3", "All")

    if show_d1 and d1_det is not None and d1_det.mean_ is not None:
        lo = d1_det.mean_ - 2.0 * d1_det.std_
        hi = d1_det.mean_ + 2.0 * d1_det.std_
        for y_val in (lo, hi):
            fig.add_shape(
                type="line",
                x0=x_lo, x1=x_hi, y0=y_val, y1=y_val,
                line=dict(color="#2ca02c", dash="dash", width=1.5),
            )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"D1 boundary (±2σ_syn = {lo:.2f}, {hi:.2f})",
            line=dict(color="#2ca02c", dash="dash"),
        ))

    if show_d2 and d2_det is not None and d2_det.threshold_ is not None:
        thr = d2_det.threshold_
        fig.add_shape(
            type="line",
            x0=thr, x1=thr, y0=y_lo, y1=y_hi,
            line=dict(color="#ff7f0e", dash="dash", width=1.5),
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"D2 boundary (H_lex = {thr:.2f})",
            line=dict(color="#ff7f0e", dash="dash"),
        ))

    if show_d3 and d3_det is not None and d3_det.sigma_inv_ is not None:
        ex, ey = _ellipse_points(d3_det.mu_, d3_det.sigma_inv_, d3_det.threshold_)
        fig.add_trace(go.Scatter(
            x=list(ex) + [ex[0]], y=list(ey) + [ey[0]],
            mode="lines",
            name=f"D3 boundary (τ = {d3_det.threshold_:.2f})",
            line=dict(color="#9467bd", dash="dash", width=2),
        ))

    fig.update_layout(height=540)
    return fig


def surprisal_bar(tokens, surprisals):
    """Bar chart of per-token surprisal (bits)."""
    fig = go.Figure(go.Bar(
        x=tokens,
        y=surprisals,
        marker_color="#7b9ecc",
        hovertemplate="%{x}: %{y:.3f} bits<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Token",
        yaxis_title="Surprisal (bits)",
        font=_FONT,
        height=320,
        margin=dict(l=55, r=20, t=25, b=60),
    )
    return fig


def dep_label_bar(label_counts):
    """Bar chart of dependency-label frequencies."""
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    fig = go.Figure(go.Bar(
        x=labels,
        y=counts,
        marker_color="#e07b54",
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Dependency label",
        yaxis_title="Count",
        font=_FONT,
        height=300,
        margin=dict(l=55, r=20, t=25, b=60),
    )
    return fig


def correlation_scatter(h_lex, h_syn, title):
    """Scatter with OLS regression line; Pearson r computed from the data."""
    h_lex = np.array(h_lex)
    h_syn = np.array(h_syn)
    r, _ = pearsonr(h_lex, h_syn)
    slope, intercept, *_ = linregress(h_lex, h_syn)
    x_line = np.linspace(h_lex.min(), h_lex.max(), 100)
    y_line = slope * x_line + intercept

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(h_lex), y=list(h_syn),
        mode="markers",
        marker=dict(color="#7b9ecc", opacity=0.12, size=3),
        hovertemplate="H_lex=%{x:.3f}<br>H_syn=%{y:.3f}<extra></extra>",
        name="Points",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=list(x_line), y=list(y_line),
        mode="lines",
        line=dict(color="#334466", width=1.5, dash="dash"),
        name="OLS",
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=f"{title}<br>Pearson r = {r:.3f}", font=dict(size=15)),
        xaxis_title="H_lex (bits)",
        yaxis_title="H_syn (bits)",
        font=_FONT,
        height=420,
        margin=dict(l=60, r=20, t=70, b=60),
    )
    return fig


def h_distribution(orig_vals, obf_vals, feature_name):
    """Overlaid histograms: original vs obfuscated phishing sentences."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=orig_vals,
        name="Original phishing",
        marker_color="#e07b54",
        opacity=0.60,
        nbinsx=40,
        histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=obf_vals,
        name="Obfuscated phishing",
        marker_color="#a84e2f",
        opacity=0.60,
        nbinsx=40,
        histnorm="probability density",
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title=f"{feature_name} (bits)",
        yaxis_title="Density",
        font=_FONT,
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=20, t=50, b=60),
    )
    return fig
