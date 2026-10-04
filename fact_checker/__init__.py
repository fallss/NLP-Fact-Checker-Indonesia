"""
Indonesian News Fact-Checker
============================

Pipeline verifikasi klaim berbahasa Indonesia:
hybrid retrieval (BM25 + Sentence-BERT + RRF) → passage selection →
NLI multilingual (mDeBERTa-v3) → agregasi multi-evidence → SHAP.

Penggunaan cepat::

    from fact_checker import FactChecker
    fc = FactChecker()
    result = fc.check("FIFA mencabut status Indonesia sebagai tuan rumah Piala Dunia U-20.")
    print(result.verdict.label, result.verdict.confidence)
"""

__version__ = "2.0.0"

from fact_checker.config import Config  # noqa: E402,F401


def __getattr__(name):
    # Impor malas: modul berat (torch/transformers) baru dimuat saat dibutuhkan
    if name == "FactChecker":
        from fact_checker.pipeline import FactChecker

        return FactChecker
    raise AttributeError(name)


__all__ = ["Config", "FactChecker", "__version__"]
