# =============================================================================
# schemas.py
# Struktur data (dataclass) yang dipertukarkan antar-modul pipeline.
# Dengan objek terstruktur, hasil fact-checking mudah ditampilkan di CLI,
# Gradio, maupun diekspor ke JSON untuk evaluasi.
# =============================================================================

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Evidence:
    """Satu passage bukti yang diambil dari artikel berita."""

    article_id: int
    title: str
    source: str
    url: str
    date: str
    passage: str
    relevance: float                 # cosine similarity klaim <-> passage (0..1)
    article_rank: int                # peringkat artikel pada hybrid retrieval
    nli_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def nli_label(self) -> str:
        if not self.nli_scores:
            return "NEUTRAL"
        return max(self.nli_scores, key=self.nli_scores.get)

    @property
    def nli_confidence(self) -> float:
        return self.nli_scores.get(self.nli_label, 0.0) if self.nli_scores else 0.0


@dataclass
class Verdict:
    """Keputusan akhir hasil agregasi multi-evidence."""

    label: str                        # SUPPORTED | REFUTED | NOT_ENOUGH_INFO
    confidence: float                 # 0..1
    label_scores: Dict[str, float]    # distribusi skor per verdict
    decisive_index: Optional[int]     # indeks evidence penentu keputusan
    reason: str                       # penjelasan singkat (bahasa Indonesia)
    conflicting: bool = False         # ada bukti kuat pro & kontra sekaligus


@dataclass
class FactCheckResult:
    """Hasil lengkap satu kali pemeriksaan klaim."""

    claim: str
    verdict: Verdict
    evidences: List[Evidence]
    explanation: Optional[List[Tuple[str, float]]] = None   # token SHAP (urutan asli)
    explanation_target: Optional[str] = None                # label NLI yang dijelaskan
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def decisive_evidence(self) -> Optional[Evidence]:
        idx = self.verdict.decisive_index
        if idx is None or not (0 <= idx < len(self.evidences)):
            return None
        return self.evidences[idx]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["decisive_evidence"] = (
            asdict(self.decisive_evidence) if self.decisive_evidence else None
        )
        return data
