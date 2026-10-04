# =============================================================================
# explainability.py
# Explainable AI: kata mana di dalam evidence yang paling memengaruhi keputusan
# model NLI?
#
# Metode utama: SHAP (SHapley Additive exPlanations) dengan Partition Explainer
# dan masker teks. Kata-kata premise di-mask secara hierarkis, lalu kontribusi
# marginal tiap kata terhadap probabilitas label target dihitung:
#
#     φ_i = Σ_{S ⊆ F\{i}}  |S|!(|F|−|S|−1)! / |F|!  · [f(S ∪ {i}) − f(S)]
#
# Fallback (bila SHAP gagal/terlalu lambat): Occlusion / leave-one-out, yaitu
# Δ probabilitas ketika satu kata dihapus.
# =============================================================================

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fact_checker.config import NLI_LABELS
from fact_checker.preprocessing import truncate_text

logger = logging.getLogger(__name__)

TokenScores = List[Tuple[str, float]]

# Warna konsisten dengan UI: hijau = mendukung label target, merah = menentang
POSITIVE_RGB = (22, 163, 74)
NEGATIVE_RGB = (220, 38, 38)


def _make_predict_fn(nli_model, hypothesis: str, target_label: str):
    target_idx = NLI_LABELS.index(target_label)

    def predict_fn(texts) -> np.ndarray:
        premises = [str(t) for t in texts]
        return nli_model.predict_proba(premises, hypothesis)[:, target_idx]

    return predict_fn


def explain_prediction(
    nli_model,
    premise: str,
    hypothesis: str,
    target_label: str = "ENTAILMENT",
    max_evals: int = 120,
    max_words: int = 80,
    method: str = "shap",
) -> Tuple[TokenScores, str]:
    """
    Menghitung kontribusi tiap kata premise terhadap P(target_label).

    Returns:
        (tokens, method_used) — tokens = [(token, nilai)] dalam URUTAN ASLI teks,
        sehingga bisa langsung dipakai untuk highlight.
    """
    premise = truncate_text(premise, max_words=max_words).rstrip(" .")
    predict_fn = _make_predict_fn(nli_model, hypothesis, target_label)

    if method == "shap":
        try:
            import shap

            masker = shap.maskers.Text(tokenizer=r"\W+")
            explainer = shap.Explainer(predict_fn, masker)
            sv = explainer([premise], max_evals=max_evals, silent=True)
            values = np.asarray(sv.values[0], dtype=float)
            if values.ndim > 1:
                values = values[:, 0]
            tokens = [(str(t), float(v)) for t, v in zip(sv.data[0], values)]
            return tokens, "shap"
        except Exception as exc:  # pragma: no cover - tergantung versi shap
            logger.warning("SHAP gagal (%s); memakai occlusion.", exc)

    return occlusion_explanation(predict_fn, premise), "occlusion"


def occlusion_explanation(predict_fn, premise: str) -> TokenScores:
    """Leave-one-word-out: skor kata = P(asli) − P(tanpa kata tersebut)."""
    parts = re.findall(r"\S+\s*", premise)
    variants = [premise] + ["".join(parts[:i] + parts[i + 1 :]) for i in range(len(parts))]
    probs = predict_fn(variants)
    base = probs[0]
    return [(tok, float(base - p)) for tok, p in zip(parts, probs[1:])]


def top_tokens(tokens: TokenScores, n: int = 10) -> TokenScores:
    """Kata dengan |kontribusi| terbesar (tanda baca & stopword pendek diabaikan)."""
    agg: dict = {}
    for tok, val in tokens:
        word = tok.strip(" \t\n.,;:!?\"'()[]“”’‘-–—")
        if len(word) < 2:
            continue
        agg[word] = agg.get(word, 0.0) + val
    return sorted(agg.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]


# -----------------------------------------------------------------------------
# Visualisasi
# -----------------------------------------------------------------------------
def highlight_html(tokens: TokenScores, target_label: str = "") -> str:
    """Teks evidence dengan latar hijau/merah sebanding kontribusi SHAP."""
    if not tokens:
        return ""
    max_abs = max((abs(v) for _, v in tokens), default=0.0) or 1.0
    spans = []
    for tok, val in tokens:
        alpha = min(1.0, abs(val) / max_abs) * 0.55
        rgb = POSITIVE_RGB if val > 0 else NEGATIVE_RGB
        text = html.escape(tok)
        if alpha < 0.04:
            spans.append(text)
            continue
        spans.append(
            f'<span title="{val:+.4f}" style="background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.2f});'
            f'border-radius:3px;padding:0 1px">{text}</span>'
        )
    return "".join(spans)


def plot_token_importance(
    tokens: TokenScores,
    target_label: str,
    top_n: int = 12,
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
):
    """Bar chart horizontal kontribusi kata; mengembalikan figure matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = top_tokens(tokens, top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * len(items) + 1.2)), dpi=130)
    if not items:
        ax.text(0.5, 0.5, "Tidak ada kontribusi kata", ha="center", va="center")
        ax.axis("off")
        return fig

    words = [w for w, _ in items]
    vals = [v for _, v in items]
    colors = ["#16a34a" if v > 0 else "#dc2626" for v in vals]
    bars = ax.barh(range(len(words)), vals, color=colors, height=0.62)
    ax.set_yticks(range(len(words)), words, fontsize=10)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    span = max(abs(v) for v in vals) or 1.0
    for bar, v in zip(bars, vals):
        ax.text(
            v + (0.02 * span if v >= 0 else -0.02 * span),
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5,
            color="#334155",
        )
    ax.set_xlim(-span * 1.3 if min(vals) < 0 else 0, span * 1.3 if max(vals) > 0 else 0.01)
    ax.set_xlabel(f"Kontribusi terhadap P({target_label})", fontsize=10)
    ax.set_title(title or f"Kata paling berpengaruh → {target_label}", fontsize=12, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info("Grafik explainability disimpan: %s", save_path)
    return fig
