# =============================================================================
# pipeline.py
# Orkestrator end-to-end: klaim -> retrieval -> NLI -> agregasi -> explainability
#
#   ┌────────────┐   ┌──────────────────────┐   ┌──────────────┐   ┌────────────┐
#   │   Klaim    │──▶│ Hybrid Retrieval     │──▶│ NLI mDeBERTa │──▶│ Agregasi   │
#   └────────────┘   │ BM25 + SBERT + RRF   │   │ (batch)      │   │ verdict    │
#                    │ → passage selection  │   └──────────────┘   └─────┬──────┘
#                    └──────────────────────┘                            ▼
#                                                              ┌────────────────┐
#                                                              │ SHAP (evidence │
#                                                              │ penentu)       │
#                                                              └────────────────┘
# =============================================================================

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from fact_checker.aggregation import aggregate
from fact_checker.config import VERDICT_NEI, VERDICT_TO_NLI, Config
from fact_checker.explainability import explain_prediction
from fact_checker.preprocessing import load_articles
from fact_checker.schemas import FactCheckResult

logger = logging.getLogger(__name__)

# Naikkan bila logika preprocessing berubah agar cache korpus dibuat ulang
PREPROCESS_VERSION = 2


def load_corpus(cfg: Config) -> pd.DataFrame:
    """Memuat korpus bersih, memakai cache pickle bila data.csv tidak berubah."""
    path = Path(cfg.data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Letakkan data.csv di root project "
            "atau gunakan opsi --data."
        )
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|{cfg.max_articles}|v{PREPROCESS_VERSION}"
    cache_file = Path(cfg.cache_dir) / f"corpus_{hashlib.sha1(key.encode()).hexdigest()[:12]}.pkl"
    if cache_file.exists():
        logger.info("Korpus dimuat dari cache: %s", cache_file.name)
        return pd.read_pickle(cache_file)

    logger.info("Memuat & membersihkan dataset: %s", path)
    df = load_articles(path, max_rows=cfg.max_articles)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache_file)
    logger.info("Korpus: %d artikel bersih (cache: %s)", len(df), cache_file.name)
    return df


class FactChecker:
    """
    Sistem fact-checking berita berbahasa Indonesia.

    Contoh:
        >>> fc = FactChecker()                       # memuat data & model
        >>> result = fc.check("Jokowi memerintahkan Wapres meninjau Plumpang.")
        >>> result.verdict.label, result.verdict.confidence
        ('SUPPORTED', 0.98)
    """

    def __init__(self, config: Optional[Config] = None, load_nli: bool = True):
        from fact_checker.nli_model import NLIModel
        from fact_checker.retrieval import DenseEncoder, HybridRetriever

        self.config = cfg = config or Config()
        t0 = time.perf_counter()

        self.articles = load_corpus(cfg)
        self.encoder = DenseEncoder(cfg.embedding_model, device=cfg.device, batch_size=cfg.embed_batch_size)
        self.retriever = HybridRetriever(
            self.articles,
            self.encoder,
            cache_dir=cfg.cache_dir,
            rrf_k=cfg.rrf_k,
            bm25_k1=cfg.bm25_k1,
            bm25_b=cfg.bm25_b,
            passage_sentences=cfg.passage_sentences,
            passage_stride=cfg.passage_stride,
        )
        self.nli = (
            NLIModel(cfg.nli_model, device=cfg.device, max_length=cfg.nli_max_length,
                     batch_size=cfg.nli_batch_size)
            if load_nli else None
        )
        self.load_seconds = time.perf_counter() - t0
        logger.info("Sistem siap dalam %.1f detik.", self.load_seconds)

    # ------------------------------------------------------------------
    def check(
        self,
        claim: str,
        explain: bool = True,
        top_passages: Optional[int] = None,
        retrieval_method: str = "hybrid",
        aggregation: Optional[str] = None,
    ) -> FactCheckResult:
        """Memeriksa satu klaim dan mengembalikan FactCheckResult."""
        if self.nli is None:
            raise RuntimeError("FactChecker dibuat dengan load_nli=False.")
        cfg = self.config
        claim = " ".join(str(claim).split())
        timings = {}

        t = time.perf_counter()
        evidences = self.retriever.retrieve(
            claim,
            top_articles=cfg.top_articles,
            top_passages=top_passages or cfg.top_passages,
            max_per_article=cfg.max_passages_per_article,
            method=retrieval_method,
        )
        timings["retrieval"] = time.perf_counter() - t

        t = time.perf_counter()
        for ev, scores in zip(evidences, self.nli.predict_batch(claim, [e.passage for e in evidences])):
            ev.nli_scores = scores
        timings["nli"] = time.perf_counter() - t

        verdict = aggregate(
            evidences,
            strategy=aggregation or cfg.aggregation,
            relevance_threshold=cfg.relevance_threshold,
            support_threshold=cfg.support_threshold,
            refute_threshold=cfg.refute_threshold,
        )
        result = FactCheckResult(claim=claim, verdict=verdict, evidences=evidences, timings=timings)

        decisive = result.decisive_evidence
        if explain and decisive is not None:
            t = time.perf_counter()
            # Jelaskan label NLI yang sesuai verdict; untuk NEI jelaskan label
            # terkuat pada evidence penentu agar tetap informatif.
            target = VERDICT_TO_NLI[verdict.label]
            if verdict.label == VERDICT_NEI:
                target = decisive.nli_label
            result.explanation, method = explain_prediction(
                self.nli, decisive.passage, claim, target_label=target,
                max_evals=cfg.shap_max_evals, max_words=cfg.shap_max_words,
            )
            result.explanation_target = target
            timings[f"explain_{method}"] = time.perf_counter() - t

        timings["total"] = sum(timings.values())
        return result
