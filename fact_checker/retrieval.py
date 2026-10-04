# =============================================================================
# retrieval.py
# Hybrid Evidence Retrieval dua tahap:
#
#   Tahap 1 - Article Retrieval (seluruh korpus):
#       * BM25 (sparse / leksikal)  : kuat untuk nama entitas, angka, istilah
#       * Dense (Sentence-BERT)     : kuat untuk parafrase & sinonim
#       * Reciprocal Rank Fusion    : menggabungkan kedua peringkat
#
#   Tahap 2 - Passage Selection (hanya artikel kandidat):
#       artikel dipecah menjadi passage (sliding window kalimat) lalu diurutkan
#       berdasarkan cosine similarity terhadap klaim.
#
# Embedding korpus di-cache ke disk (folder cache/) sehingga hanya dihitung sekali.
# =============================================================================

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from fact_checker.preprocessing import make_passages
from fact_checker.schemas import Evidence

logger = logging.getLogger(__name__)

# Stopword bahasa Indonesia (ringkas) untuk BM25
INDONESIAN_STOPWORDS = frozenset(
    """
    yang dan di ke dari untuk pada dengan ini itu dalam tidak akan juga atau ada
    oleh sebagai adalah karena bahwa telah sudah masih bisa dapat saat para kata
    lebih ia mereka kami kita saya anda sebuah secara hingga agar namun tersebut
    menjadi jika serta antara setelah sejak bagi tak pun lagi baru hal kepada
    seperti maupun belum harus lalu yakni yaitu tetapi tapi sedang sangat hanya
    semua banyak dua satu tiga pihak nya pula dia sendiri tanpa terhadap begitu
    kalau ketika sementara bila mana apa siapa bagaimana mengapa kapan dimana
    """.split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Tokenisasi sederhana untuk BM25: lowercase, alfanumerik, tanpa stopword."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in INDONESIAN_STOPWORDS
    ]


# =============================================================================
# BM25 (implementasi vektor dengan matriks sparse -> cepat untuk 30 ribu+ dokumen)
# =============================================================================
class BM25Index:
    """
    Okapi BM25:
        score(q, d) = Σ_t∈q  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
        IDF(t)      = ln( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )
    Bobot per (dokumen, term) dihitung sekali saat fit(), sehingga skor query
    cukup berupa penjumlahan kolom matriks sparse.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab: Dict[str, int] = {}
        self.weights: Optional[sparse.csc_matrix] = None

    def fit(self, documents: Sequence[str]) -> "BM25Index":
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            tokenizer=tokenize, lowercase=False, token_pattern=None, dtype=np.float32
        )
        tf = vectorizer.fit_transform(documents).tocsr().astype(np.float32)
        self.vocab = vectorizer.vocabulary_

        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = float(doc_len.mean()) if n_docs else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1]).astype(np.float32)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        # Normalisasi panjang dokumen untuk setiap entri non-nol
        norm = self.k1 * (1 - self.b + self.b * doc_len / avgdl)
        row_idx = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        data = tf.data
        tf.data = (data * (self.k1 + 1) / (data + norm[row_idx])) * idf[tf.indices]
        self.weights = tf.tocsc()
        return self

    def score(self, query: str) -> np.ndarray:
        """Skor BM25 query terhadap seluruh dokumen."""
        if self.weights is None:
            raise RuntimeError("BM25Index belum di-fit.")
        term_ids = [self.vocab[t] for t in set(tokenize(query)) if t in self.vocab]
        if not term_ids:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel()


# =============================================================================
# Dense Retrieval (Sentence-BERT) dengan cache embedding
# =============================================================================
class DenseEncoder:
    """Pembungkus SentenceTransformer dengan embedding ter-normalisasi (cosine = dot)."""

    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        logger.info("Memuat model embedding: %s", model_name)
        self.model_name = model_name
        try:  # utamakan cache lokal -> cepat & bisa offline
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)


def _fingerprint(model_name: str, texts: Sequence[str]) -> str:
    """Hash isi korpus + nama model -> nama file cache yang unik."""
    h = hashlib.sha1(model_name.encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def load_or_compute_embeddings(
    encoder: DenseEncoder,
    texts: Sequence[str],
    cache_dir: Optional[Path],
    tag: str = "articles",
) -> np.ndarray:
    """Memuat embedding dari cache .npy, atau menghitung & menyimpannya."""
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^A-Za-z0-9]+", "-", encoder.model_name).strip("-")
        cache_file = cache_dir / f"emb_{tag}_{safe_model}_{_fingerprint(encoder.model_name, texts)}.npy"
        if cache_file.exists():
            emb = np.load(cache_file)
            if emb.shape[0] == len(texts):
                logger.info("Embedding dimuat dari cache: %s", cache_file.name)
                return emb

    logger.warning(
        "Menghitung embedding %d dokumen — BUKAN download. Hanya dilakukan sekali "
        "(±15 menit di CPU), hasilnya disimpan ke folder cache/.", len(texts)
    )
    emb = encoder.encode(texts, show_progress=True)
    if cache_file is not None:
        np.save(cache_file, emb)
        logger.info("Embedding disimpan ke cache: %s", cache_file)
    return emb


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> Dict[int, float]:
    """
    RRF (Cormack et al., 2009):  RRF(d) = Σ_r 1 / (k + rank_r(d))
    Menggabungkan beberapa daftar peringkat tanpa perlu menyamakan skala skor.
    """
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


# =============================================================================
# Hybrid Retriever
# =============================================================================
class HybridRetriever:
    """Retriever dua tahap (artikel -> passage) untuk korpus berita."""

    def __init__(
        self,
        articles: pd.DataFrame,
        encoder: DenseEncoder,
        cache_dir: Optional[Path] = None,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        passage_sentences: int = 3,
        passage_stride: int = 2,
        candidate_pool: int = 100,
    ):
        self.articles = articles.reset_index(drop=True)
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.passage_sentences = passage_sentences
        self.passage_stride = passage_stride
        self.candidate_pool = candidate_pool
        self._passage_cache: Dict[int, List[str]] = {}

        # BM25 atas judul + isi artikel
        logger.info("Membangun indeks BM25 untuk %d artikel...", len(self.articles))
        bm25_docs = (self.articles["title"] + " " + self.articles["content"]).tolist()
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b).fit(bm25_docs)

        # Dense atas judul + ringkasan (atau awal konten jika ringkasan kosong).
        # Model MiniLM hanya membaca 128 token pertama, sehingga representasi
        # ringkas "judul + ringkasan" jauh lebih informatif daripada awal artikel.
        dense_docs = [
            f"{row.title}. {row.summary if len(str(row.summary).split()) >= 10 else row.content[:600]}"
            for row in self.articles.itertuples()
        ]
        self.article_embeddings = load_or_compute_embeddings(
            encoder, dense_docs, cache_dir, tag=f"articles{len(dense_docs)}"
        )

    # ------------------------------------------------------------------ tahap 1
    def search_articles(
        self, claim: str, top_k: int = 8, method: str = "hybrid",
        claim_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float]]:
        """
        Mengembalikan daftar (indeks_artikel, skor) terurut.
        method: "hybrid" (RRF), "bm25", atau "dense".
        """
        pool = min(self.candidate_pool, len(self.articles))
        rankings = []
        if method in ("hybrid", "bm25"):
            bm25_scores = self.bm25.score(claim)
            rankings.append(_top_indices(bm25_scores, pool))
            if method == "bm25":
                return [(int(i), float(bm25_scores[i])) for i in rankings[0][:top_k]]
        if method in ("hybrid", "dense"):
            q = claim_embedding if claim_embedding is not None else self.encoder.encode([claim])[0]
            dense_scores = self.article_embeddings @ q
            rankings.append(_top_indices(dense_scores, pool))
            if method == "dense":
                return [(int(i), float(dense_scores[i])) for i in rankings[0][:top_k]]
        if method != "hybrid":
            raise ValueError(f"Metode retrieval tidak dikenal: {method}")

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(int(i), float(s)) for i, s in ranked]

    # ------------------------------------------------------------------ tahap 2
    def passages_of(self, article_idx: int) -> List[str]:
        if article_idx not in self._passage_cache:
            self._passage_cache[article_idx] = make_passages(
                self.articles.at[article_idx, "content"],
                sentences_per_passage=self.passage_sentences,
                stride=self.passage_stride,
            )
        return self._passage_cache[article_idx]

    def retrieve(
        self,
        claim: str,
        top_articles: int = 8,
        top_passages: int = 5,
        max_per_article: int = 2,
        method: str = "hybrid",
    ) -> List[Evidence]:
        """Pipeline retrieval lengkap: klaim -> daftar Evidence (passage)."""
        claim_emb = self.encoder.encode([claim])[0]
        hits = self.search_articles(claim, top_articles, method, claim_embedding=claim_emb)

        candidates: List[Tuple[int, int, str]] = []  # (article_idx, article_rank, passage)
        for rank, (art_idx, _) in enumerate(hits, start=1):
            for passage in self.passages_of(art_idx):
                candidates.append((art_idx, rank, passage))
        if not candidates:
            return []

        passage_emb = self.encoder.encode([c[2] for c in candidates])
        sims = passage_emb @ claim_emb

        evidences: List[Evidence] = []
        per_article: Dict[int, int] = {}
        seen_passages = set()
        for i in np.argsort(-sims):
            art_idx, art_rank, passage = candidates[i]
            if per_article.get(art_idx, 0) >= max_per_article or passage in seen_passages:
                continue
            row = self.articles.iloc[art_idx]
            evidences.append(
                Evidence(
                    article_id=int(row["id"]),
                    title=row["title"],
                    source=row["source"],
                    url=row["url"],
                    date=row["date"],
                    passage=passage,
                    relevance=float(max(0.0, sims[i])),
                    article_rank=art_rank,
                )
            )
            per_article[art_idx] = per_article.get(art_idx, 0) + 1
            seen_passages.add(passage)
            if len(evidences) >= top_passages:
                break
        return evidences


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indeks k skor tertinggi (terurut menurun) dengan argpartition O(n)."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]
