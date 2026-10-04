# =============================================================================
# aggregation.py
# Menggabungkan hasil NLI dari banyak evidence menjadi SATU verdict.
#
# Strategi:
#   "max"      (default) : gaya FEVER — cukup satu passage relevan yang secara
#                          meyakinkan mendukung/membantah klaim. Passage lain yang
#                          netral (membahas hal berbeda) tidak "mengencerkan" bukti.
#   "weighted"           : rata-rata probabilitas berbobot relevansi (similarity²).
#   "mean"               : rata-rata biasa semua evidence (strategi versi lama,
#                          dipertahankan untuk studi ablasi).
#
# Gerbang relevansi: evidence dengan cosine similarity < relevance_threshold
# dianggap tidak membahas klaim, sehingga tidak boleh menentukan verdict.
# =============================================================================

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from fact_checker.config import (
    NLI_TO_VERDICT,
    VERDICT_NEI,
    VERDICT_REFUTED,
    VERDICT_SUPPORTED,
)
from fact_checker.schemas import Evidence, Verdict


def aggregate(
    evidences: Sequence[Evidence],
    strategy: str = "max",
    relevance_threshold: float = 0.35,
    support_threshold: float = 0.60,
    refute_threshold: float = 0.60,
) -> Verdict:
    """Menghasilkan Verdict dari evidence yang sudah memiliki nli_scores."""
    if not evidences:
        return Verdict(
            label=VERDICT_NEI,
            confidence=1.0,
            label_scores={VERDICT_SUPPORTED: 0.0, VERDICT_REFUTED: 0.0, VERDICT_NEI: 1.0},
            decisive_index=None,
            reason="Tidak ada artikel berita yang ditemukan untuk klaim ini.",
        )

    ent = np.array([e.nli_scores.get("ENTAILMENT", 0.0) for e in evidences])
    con = np.array([e.nli_scores.get("CONTRADICTION", 0.0) for e in evidences])
    neu = np.array([e.nli_scores.get("NEUTRAL", 0.0) for e in evidences])
    rel = np.array([e.relevance for e in evidences])
    relevant = rel >= relevance_threshold

    if strategy == "max":
        return _aggregate_max(
            ent, con, rel, relevant, relevance_threshold, support_threshold, refute_threshold
        )
    if strategy in ("weighted", "mean"):
        mask = relevant if (strategy == "weighted" and relevant.any()) else np.ones_like(relevant)
        weights = (rel ** 2) * mask if strategy == "weighted" else mask.astype(float)
        if weights.sum() <= 0:
            weights = np.ones_like(rel)
        weights = weights / weights.sum()
        avg = {
            "ENTAILMENT": float(weights @ ent),
            "NEUTRAL": float(weights @ neu),
            "CONTRADICTION": float(weights @ con),
        }
        best_nli = max(avg, key=avg.get)
        label = NLI_TO_VERDICT[best_nli]
        column = {"ENTAILMENT": ent, "CONTRADICTION": con, "NEUTRAL": rel}[best_nli]
        return Verdict(
            label=label,
            confidence=avg[best_nli],
            label_scores={NLI_TO_VERDICT[k]: v for k, v in avg.items()},
            decisive_index=int(np.argmax(column * weights)),
            reason=f"Rata-rata {'berbobot ' if strategy == 'weighted' else ''}"
            f"probabilitas NLI dari {len(evidences)} evidence.",
        )
    raise ValueError(f"Strategi agregasi tidak dikenal: {strategy}")


def _aggregate_max(
    ent: np.ndarray,
    con: np.ndarray,
    rel: np.ndarray,
    relevant: np.ndarray,
    relevance_threshold: float,
    support_threshold: float,
    refute_threshold: float,
) -> Verdict:
    if not relevant.any():
        top = int(np.argmax(rel))
        return Verdict(
            label=VERDICT_NEI,
            confidence=float(1.0 - rel[top]),
            label_scores={VERDICT_SUPPORTED: 0.0, VERDICT_REFUTED: 0.0, VERDICT_NEI: float(1.0 - rel[top])},
            decisive_index=top,
            reason=(
                f"Tidak ada evidence yang cukup relevan (similarity tertinggi "
                f"{rel[top]:.0%} < ambang {relevance_threshold:.0%})."
            ),
        )

    ent_r = np.where(relevant, ent, 0.0)
    con_r = np.where(relevant, con, 0.0)
    i_sup, i_ref = int(np.argmax(ent_r)), int(np.argmax(con_r))
    support, refute = float(ent_r[i_sup]), float(con_r[i_ref])
    nei = 1.0 - max(support, refute)
    scores = {VERDICT_SUPPORTED: support, VERDICT_REFUTED: refute, VERDICT_NEI: nei}
    n_rel = int(relevant.sum())

    strong_sup = support >= support_threshold
    strong_ref = refute >= refute_threshold
    conflicting = strong_sup and strong_ref

    if strong_sup and (support >= refute):
        reason = f"Evidence #{i_sup + 1} mendukung klaim (P(entailment) = {support:.0%})."
        if conflicting:
            reason += f" Namun evidence #{i_ref + 1} membantahnya ({refute:.0%}) — bukti saling bertentangan."
        return Verdict(VERDICT_SUPPORTED, support, scores, i_sup, reason, conflicting)
    if strong_ref:
        reason = f"Evidence #{i_ref + 1} bertentangan dengan klaim (P(contradiction) = {refute:.0%})."
        if conflicting:
            reason += f" Namun evidence #{i_sup + 1} mendukungnya ({support:.0%}) — bukti saling bertentangan."
        return Verdict(VERDICT_REFUTED, refute, scores, i_ref, reason, conflicting)

    top = int(np.argmax(np.where(relevant, rel, -1.0)))
    return Verdict(
        label=VERDICT_NEI,
        confidence=nei,
        label_scores=scores,
        decisive_index=top,
        reason=(
            f"{n_rel} evidence relevan ditemukan, tetapi tidak ada yang secara meyakinkan "
            f"mendukung (maks {support:.0%}) atau membantah (maks {refute:.0%}) klaim."
        ),
    )


def verdict_explanation(label: str) -> str:
    """Penjelasan naratif tiap verdict untuk ditampilkan ke pengguna."""
    return {
        VERDICT_SUPPORTED: (
            "Sistem menemukan passage berita yang isinya konsisten dan mengimplikasikan "
            "kebenaran klaim (entailment)."
        ),
        VERDICT_REFUTED: (
            "Sistem menemukan passage berita yang fakta-faktanya bertentangan dengan klaim "
            "(contradiction). Klaim berpotensi hoaks atau keliru."
        ),
        VERDICT_NEI: (
            "Berita yang tersedia tidak cukup untuk membuktikan maupun membantah klaim. "
            "Klaim mungkin membahas detail yang tidak diberitakan atau di luar cakupan korpus."
        ),
    }.get(label, "")
