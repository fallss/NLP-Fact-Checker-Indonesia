# =============================================================================
# evaluation.py
# Evaluasi kuantitatif sistem pada benchmark berlabel (data/benchmark.jsonl).
#
# Tiga level evaluasi:
#   1. NLI (oracle evidence)  : klaim + evidence emas -> label NLI
#                               => mengukur kualitas model NLI secara terisolasi
#   2. Retrieval              : apakah artikel emas terambil? Recall@k & MRR
#                               (dibandingkan: BM25 vs Dense vs Hybrid RRF)
#   3. End-to-end             : klaim saja -> verdict akhir sistem
#                               (+ ablasi strategi agregasi)
#
# Catatan perbaikan: versi lama memanggil classification_report tanpa argumen
# `labels`, sehingga baris laporan tertukar (sklearn mengurutkan label secara
# alfabetis). Di sini urutan label selalu diberikan secara eksplisit.
# =============================================================================

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from fact_checker.aggregation import aggregate
from fact_checker.config import NLI_LABELS, NLI_TO_VERDICT, PROJECT_ROOT, VERDICTS

logger = logging.getLogger(__name__)

BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark.jsonl"


@dataclass
class BenchmarkCase:
    id: int
    topic: str
    claim: str
    label: str                      # label NLI emas
    evidence: Optional[str]
    article_id: Optional[int]
    gold_article_ids: List[int]

    @property
    def verdict(self) -> str:
        return NLI_TO_VERDICT[self.label]


def load_benchmark(path: Path | str = BENCHMARK_PATH) -> List[BenchmarkCase]:
    with open(path, encoding="utf-8") as f:
        return [BenchmarkCase(**json.loads(line)) for line in f if line.strip()]


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict:
    """Accuracy, macro P/R/F1, laporan per kelas, dan confusion matrix."""
    labels = list(labels)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "report_text": classification_report(
            y_true, y_pred, labels=labels, digits=3, zero_division=0
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


# -----------------------------------------------------------------------------
# 1. NLI dengan evidence emas
# -----------------------------------------------------------------------------
def evaluate_nli(nli_model, cases: Sequence[BenchmarkCase]) -> Dict:
    cases = [c for c in cases if c.evidence]
    t = time.perf_counter()
    probs = nli_model.predict_proba([c.evidence for c in cases], [c.claim for c in cases])
    elapsed = time.perf_counter() - t
    y_pred = [NLI_LABELS[i] for i in probs.argmax(axis=1)]
    y_true = [c.label for c in cases]
    metrics = classification_metrics(y_true, y_pred, NLI_LABELS)
    metrics["model"] = nli_model.model_name
    metrics["seconds_per_pair"] = elapsed / max(1, len(cases))
    metrics["predictions"] = [
        {"id": c.id, "topic": c.topic, "claim": c.claim, "gold": c.label, "pred": p,
         "confidence": float(pr.max())}
        for c, p, pr in zip(cases, y_pred, probs)
    ]
    return metrics


# -----------------------------------------------------------------------------
# 2. Retrieval
# -----------------------------------------------------------------------------
def evaluate_retrieval(
    retriever, cases: Sequence[BenchmarkCase], ks=(1, 3, 5, 10),
    methods=("bm25", "dense", "hybrid"),
) -> Dict[str, Dict[str, float]]:
    """Recall@k dan MRR artikel emas untuk tiap metode retrieval."""
    cases = [c for c in cases if c.gold_article_ids]
    id_of = retriever.articles["id"].to_numpy()
    results = {}
    for method in methods:
        hits = {k: 0 for k in ks}
        rr = []
        for c in cases:
            ranked = [int(id_of[i]) for i, _ in retriever.search_articles(c.claim, max(ks), method)]
            rank = next((r for r, aid in enumerate(ranked, 1) if aid in c.gold_article_ids), None)
            rr.append(1.0 / rank if rank else 0.0)
            for k in ks:
                hits[k] += int(rank is not None and rank <= k)
        results[method] = {f"recall@{k}": hits[k] / len(cases) for k in ks}
        results[method]["mrr"] = float(np.mean(rr))
    return results


# -----------------------------------------------------------------------------
# 3. End-to-end
# -----------------------------------------------------------------------------
def run_pipeline(fact_checker, cases: Sequence[BenchmarkCase]):
    """Menjalankan pipeline (tanpa SHAP) untuk setiap klaim -> (runs, detik/klaim)."""
    t = time.perf_counter()
    runs = [(c, fact_checker.check(c.claim, explain=False)) for c in cases]
    return runs, (time.perf_counter() - t) / max(1, len(cases))


def evaluate_end_to_end(
    fact_checker, cases: Sequence[BenchmarkCase],
    strategies=("max", "weighted", "mean"),
    runs=None, seconds_per_claim: Optional[float] = None,
) -> Dict:
    """
    Menghitung metrik verdict end-to-end. Hasil NLI per evidence dipakai ulang
    untuk ablasi strategi agregasi sehingga tidak perlu inferensi ulang.
    `runs` hasil run_pipeline() boleh diberikan agar pipeline tidak dijalankan lagi.
    """
    cfg = fact_checker.config
    if runs is None:
        runs, seconds_per_claim = run_pipeline(fact_checker, cases)

    out = {"seconds_per_claim": seconds_per_claim or 0.0, "strategies": {}}
    for strategy in strategies:
        y_true, y_pred, details = [], [], []
        for c, res in runs:
            v = aggregate(res.evidences, strategy=strategy,
                          relevance_threshold=cfg.relevance_threshold,
                          support_threshold=cfg.support_threshold,
                          refute_threshold=cfg.refute_threshold)
            y_true.append(c.verdict)
            y_pred.append(v.label)
            details.append({
                "id": c.id, "topic": c.topic, "claim": c.claim, "gold": c.verdict,
                "pred": v.label, "confidence": v.confidence,
                "evidence_hit": any(e.article_id in c.gold_article_ids for e in res.evidences),
                "top_evidence": res.evidences[v.decisive_index].title
                if v.decisive_index is not None else None,
            })
        m = classification_metrics(y_true, y_pred, VERDICTS)
        m["predictions"] = details
        out["strategies"][strategy] = m
    gold_cases = [d for d in out["strategies"][strategies[0]]["predictions"]
                  if next(c for c in cases if c.id == d["id"]).gold_article_ids]
    out["evidence_recall"] = float(np.mean([d["evidence_hit"] for d in gold_cases])) if gold_cases else 0.0
    return out


# -----------------------------------------------------------------------------
# Visualisasi & laporan
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm, labels, title: str, save_path: Optional[Path] = None, ax=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
    short = [l.replace("NOT_ENOUGH_INFO", "NEI").title() for l in labels]
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), short, fontsize=9)
    ax.set_yticks(range(len(labels)), short, fontsize=9)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label emas")
    ax.set_title(title, fontsize=11, fontweight="bold")
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thresh else "#0f172a", fontweight="bold")
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig
    return ax


def save_report(results: Dict, reports_dir: Path) -> Dict[str, Path]:
    """Menyimpan hasil evaluasi ke JSON, Markdown, dan PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def strip(obj):
        if isinstance(obj, dict):
            return {k: strip(v) for k, v in obj.items() if k != "report_text"}
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    paths["json"] = reports_dir / "evaluation_results.json"
    paths["json"].write_text(json.dumps(strip(results), indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Gambar confusion matrix ----
    panels = []
    for key, m in results.get("nli", {}).items():
        panels.append((f"NLI · {key}", m))
    if "end_to_end" in results:
        panels.append(("End-to-end (max)", results["end_to_end"]["strategies"]["max"]))
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), dpi=130)
        axes = np.atleast_1d(axes)
        for ax, (title, m) in zip(axes, panels):
            plot_confusion_matrix(m["confusion_matrix"], m["labels"],
                                  f"{title}\nacc {m['accuracy']:.1%} · F1 {m['macro_f1']:.3f}", ax=ax)
        fig.tight_layout()
        paths["confusion"] = reports_dir / "confusion_matrices.png"
        fig.savefig(paths["confusion"], bbox_inches="tight")
        plt.close(fig)

    # ---- Ringkasan Markdown ----
    lines = ["# Hasil Evaluasi Fact-Checker", ""]
    if "nli" in results:
        lines += ["## 1. NLI dengan evidence emas", "",
                  "| Model | Accuracy | Macro-P | Macro-R | Macro-F1 | detik/pasangan |",
                  "|---|---|---|---|---|---|"]
        for key, m in results["nli"].items():
            lines.append(f"| `{m['model']}` | {m['accuracy']:.1%} | {m['macro_precision']:.3f} | "
                         f"{m['macro_recall']:.3f} | **{m['macro_f1']:.3f}** | {m['seconds_per_pair']:.3f} |")
        for key, m in results["nli"].items():
            lines += ["", f"<details><summary>Laporan per kelas — {key}</summary>", "",
                      "```", m["report_text"].rstrip(), "```", "</details>"]
        lines.append("")
    if "retrieval" in results:
        ks = [k for k in next(iter(results["retrieval"].values())) if k.startswith("recall")]
        lines += ["## 2. Retrieval artikel emas", "",
                  "| Metode | " + " | ".join(k.title() for k in ks) + " | MRR |",
                  "|---|" + "---|" * (len(ks) + 1)]
        for method, m in results["retrieval"].items():
            lines.append(f"| {method} | " + " | ".join(f"{m[k]:.1%}" for k in ks) + f" | {m['mrr']:.3f} |")
        lines.append("")
    if "end_to_end" in results:
        e2e = results["end_to_end"]
        lines += ["## 3. End-to-end (klaim → verdict)", "",
                  f"Waktu rata-rata: {e2e['seconds_per_claim']:.2f} detik/klaim · "
                  f"evidence emas terambil: {e2e['evidence_recall']:.1%}", "",
                  "| Strategi agregasi | Accuracy | Macro-F1 |", "|---|---|---|"]
        for s, m in e2e["strategies"].items():
            lines.append(f"| {s} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} |")
        lines += ["", "```", e2e["strategies"]["max"]["report_text"].rstrip(), "```", ""]
        lines += ["### Kesalahan prediksi (strategi max)", "",
                  "| # | Klaim | Emas | Prediksi |", "|---|---|---|---|"]
        for d in e2e["strategies"]["max"]["predictions"]:
            if d["gold"] != d["pred"]:
                lines.append(f"| {d['id']} | {d['claim']} | {d['gold']} | {d['pred']} |")
    paths["markdown"] = reports_dir / "evaluation_summary.md"
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
