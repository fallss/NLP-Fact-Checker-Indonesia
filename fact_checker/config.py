# =============================================================================
# config.py
# Konfigurasi terpusat untuk seluruh pipeline Fact-Checker.
#
# Semua "angka ajaib" (threshold, jumlah evidence, nama model, dsb.) dikumpulkan
# di sini agar eksperimen mudah dilakukan tanpa mengubah kode modul lain.
# Nilai dapat di-override lewat argumen CLI atau variabel lingkungan FC_*.
# =============================================================================

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Optional

# Folder root project (satu tingkat di atas folder fact_checker/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Model embedding multilingual (mendukung bahasa Indonesia) untuk semantic search.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Model NLI multilingual: mDeBERTa-v3 yang di-fine-tune pada XNLI +
# multilingual-NLI-26lang-2mil7 (2,7 juta pasangan NLI, termasuk bahasa Indonesia).
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

# Model baseline versi lama (hanya dilatih pada data bahasa Inggris SNLI/MNLI).
# Disimpan untuk keperluan studi ablasi / perbandingan di laporan.
NLI_MODEL_BASELINE = "cross-encoder/nli-MiniLM2-L6-H768"

# Label NLI standar yang dipakai di seluruh project
NLI_LABELS = ("ENTAILMENT", "NEUTRAL", "CONTRADICTION")

# Verdict akhir fact-checking (gaya FEVER) beserta padanan label NLI-nya
VERDICT_SUPPORTED = "SUPPORTED"
VERDICT_REFUTED = "REFUTED"
VERDICT_NEI = "NOT_ENOUGH_INFO"
VERDICTS = (VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_NEI)

VERDICT_TO_NLI = {
    VERDICT_SUPPORTED: "ENTAILMENT",
    VERDICT_REFUTED: "CONTRADICTION",
    VERDICT_NEI: "NEUTRAL",
}
NLI_TO_VERDICT = {v: k for k, v in VERDICT_TO_NLI.items()}

VERDICT_DISPLAY = {
    VERDICT_SUPPORTED: "DIDUKUNG FAKTA",
    VERDICT_REFUTED: "BERTENTANGAN / HOAKS",
    VERDICT_NEI: "BUKTI TIDAK CUKUP",
}


@dataclass
class Config:
    """Seluruh hyper-parameter pipeline fact-checking."""

    # ---- Data ----
    data_path: Path = PROJECT_ROOT / "data.csv"
    max_articles: Optional[int] = None  # None = seluruh dataset (~32 ribu artikel)
    cache_dir: Path = PROJECT_ROOT / "cache"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # ---- Model ----
    embedding_model: str = EMBEDDING_MODEL
    nli_model: str = NLI_MODEL
    device: Optional[str] = None  # None = otomatis (cuda jika tersedia)
    embed_batch_size: int = 64
    nli_batch_size: int = 8
    nli_max_length: int = 384

    # ---- Retrieval (tahap 1: artikel, tahap 2: passage) ----
    top_articles: int = 8          # jumlah artikel kandidat hasil hybrid retrieval
    top_passages: int = 5          # jumlah passage yang diverifikasi oleh NLI
    passage_sentences: int = 3     # jumlah kalimat per passage (sliding window)
    passage_stride: int = 2        # pergeseran window (overlap = sentences - stride)
    max_passages_per_article: int = 2  # jaga keberagaman sumber bukti
    rrf_k: int = 60                # konstanta Reciprocal Rank Fusion
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ---- Agregasi verdict ----
    aggregation: str = "max"       # "max" | "weighted" | "mean"
    relevance_threshold: float = 0.35   # cosine minimal agar passage dianggap relevan
    support_threshold: float = 0.60     # P(entailment) minimal untuk verdict SUPPORTED
    refute_threshold: float = 0.60      # P(contradiction) minimal untuk verdict REFUTED

    # ---- Explainability ----
    shap_max_evals: int = 120
    shap_max_words: int = 80

    def with_overrides(self, **kwargs) -> "Config":
        """Salinan config dengan sebagian nilai diganti (nilai None diabaikan)."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)

    @classmethod
    def from_env(cls) -> "Config":
        """Membaca override dari variabel lingkungan, mis. FC_TOP_PASSAGES=3."""
        cfg = cls()
        overrides = {}
        for f in fields(cls):
            raw = os.environ.get(f"FC_{f.name.upper()}")
            if raw is None:
                continue
            current = getattr(cfg, f.name)
            if isinstance(current, bool):
                overrides[f.name] = raw.lower() in ("1", "true", "yes")
            elif isinstance(current, int):
                overrides[f.name] = int(raw)
            elif isinstance(current, float):
                overrides[f.name] = float(raw)
            elif isinstance(current, Path):
                overrides[f.name] = Path(raw)
            elif f.name == "max_articles":
                overrides[f.name] = None if raw.lower() in ("", "none", "all") else int(raw)
            else:
                overrides[f.name] = raw
        return replace(cfg, **overrides)


def resolve_device(device: Optional[str] = None) -> str:
    """Memilih perangkat komputasi: cuda > mps > cpu."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
