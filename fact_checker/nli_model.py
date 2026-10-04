# =============================================================================
# nli_model.py
# Natural Language Inference (NLI) untuk verifikasi klaim.
#
#   Premise    = passage evidence dari berita
#   Hypothesis = klaim pengguna
#
#   ENTAILMENT    -> evidence mendukung klaim
#   CONTRADICTION -> evidence bertentangan dengan klaim
#   NEUTRAL       -> evidence tidak cukup untuk memutuskan
#
# Model default: mDeBERTa-v3-base (multilingual) yang di-fine-tune pada XNLI dan
# multilingual-NLI-26lang-2mil7 — dataset NLI 27 bahasa yang mencakup bahasa
# Indonesia. Versi lama project memakai cross-encoder/nli-MiniLM2-L6-H768 yang
# hanya dilatih pada data bahasa Inggris (lihat evaluasi ablasi di laporan).
# =============================================================================

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fact_checker.config import NLI_LABELS, NLI_MODEL, NLI_MODEL_BASELINE, resolve_device

logger = logging.getLogger(__name__)

LABEL_DISPLAY = {
    "ENTAILMENT": "DIDUKUNG (ENTAILMENT)",
    "NEUTRAL": "TIDAK DAPAT DITENTUKAN (NEUTRAL)",
    "CONTRADICTION": "BERTENTANGAN (CONTRADICTION)",
}


def normalize_label(label: str) -> str:
    """Menyeragamkan nama label dari berbagai model ke ENTAILMENT/NEUTRAL/CONTRADICTION."""
    label = str(label).upper()
    if "ENTAIL" in label:
        return "ENTAILMENT"
    if "CONTRA" in label:
        return "CONTRADICTION"
    if "NEUTRAL" in label:
        return "NEUTRAL"
    raise ValueError(f"Label NLI tidak dikenali: {label!r}")


class NLIModel:
    """Cross-encoder NLI dengan inferensi batch."""

    def __init__(
        self,
        model_name: str = NLI_MODEL,
        device: Optional[str] = None,
        max_length: int = 384,
        batch_size: int = 8,
        allow_fallback: bool = True,
    ):
        self.device = resolve_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        try:
            self._load(model_name)
        except Exception as exc:  # jaringan putus / model tidak ditemukan
            if not allow_fallback or model_name == NLI_MODEL_BASELINE:
                raise
            logger.warning("Gagal memuat %s (%s). Beralih ke %s.", model_name, exc, NLI_MODEL_BASELINE)
            self._load(NLI_MODEL_BASELINE)

    def _load(self, model_name: str) -> None:
        logger.info("Memuat model NLI: %s (device=%s)", model_name, self.device)
        self.model_name = model_name
        try:  # utamakan cache lokal -> cepat & bisa offline
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Petakan indeks output model -> urutan standar NLI_LABELS
        id2label = {int(k): normalize_label(v) for k, v in self.model.config.id2label.items()}
        if sorted(id2label.values()) != sorted(NLI_LABELS):
            raise ValueError(f"Model {model_name} bukan model NLI 3-kelas: {id2label}")
        self._order = [next(i for i, l in id2label.items() if l == lbl) for lbl in NLI_LABELS]

    # ------------------------------------------------------------------ inferensi
    @torch.inference_mode()
    def predict_proba(
        self,
        premises: Sequence[str],
        hypotheses: Union[str, Sequence[str]],
    ) -> np.ndarray:
        """
        Probabilitas NLI untuk banyak pasangan sekaligus.

        Returns:
            array (n, 3) dengan urutan kolom = NLI_LABELS
        """
        premises = [str(p) for p in premises]
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses] * len(premises)
        if len(premises) != len(hypotheses):
            raise ValueError("Jumlah premise dan hypothesis harus sama.")
        if not premises:
            return np.zeros((0, len(NLI_LABELS)), dtype=np.float32)

        # Urutkan berdasarkan panjang agar padding per-batch minimal
        order = np.argsort([len(p) for p in premises])
        out = np.zeros((len(premises), len(NLI_LABELS)), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer(
                [premises[i] for i in idx],
                [hypotheses[i] for i in idx],
                return_tensors="pt",
                padding=True,
                truncation="only_first",   # klaim tidak pernah dipotong
                max_length=self.max_length,
            ).to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            out[idx] = probs[:, self._order]
        return out

    def predict(self, premise: str, hypothesis: str) -> Tuple[str, float, Dict[str, float]]:
        """Prediksi satu pasangan -> (label, confidence, skor semua label)."""
        probs = self.predict_proba([premise], hypothesis)[0]
        scores = {lbl: float(p) for lbl, p in zip(NLI_LABELS, probs)}
        label = max(scores, key=scores.get)
        return label, scores[label], scores

    def predict_batch(self, claim: str, evidences: Sequence[str]) -> List[Dict[str, float]]:
        """Satu klaim terhadap banyak evidence -> list skor per evidence."""
        probs = self.predict_proba(evidences, claim)
        return [{lbl: float(p) for lbl, p in zip(NLI_LABELS, row)} for row in probs]

    @staticmethod
    def get_label_display(label: str) -> str:
        return LABEL_DISPLAY.get(label, label)
