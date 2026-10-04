from fact_checker.aggregation import aggregate
from fact_checker.config import VERDICT_NEI, VERDICT_REFUTED, VERDICT_SUPPORTED
from fact_checker.schemas import Evidence


def ev(rel, e, n, c):
    return Evidence(article_id=0, title="t", source="s", url="", date="", passage="p",
                    relevance=rel, article_rank=1,
                    nli_scores={"ENTAILMENT": e, "NEUTRAL": n, "CONTRADICTION": c})


def test_single_strong_support_wins_with_max_strategy():
    evs = [ev(0.4, 0.05, 0.9, 0.05), ev(0.7, 0.92, 0.05, 0.03), ev(0.5, 0.1, 0.85, 0.05)]
    v = aggregate(evs, strategy="max")
    assert v.label == VERDICT_SUPPORTED
    assert v.decisive_index == 1
    assert abs(v.confidence - 0.92) < 1e-9


def test_mean_strategy_dilutes_evidence():
    # Kelemahan strategi lama: satu bukti kuat "tenggelam" oleh evidence netral
    evs = [ev(0.4, 0.05, 0.9, 0.05), ev(0.7, 0.92, 0.05, 0.03), ev(0.5, 0.1, 0.85, 0.05)]
    assert aggregate(evs, strategy="mean").label == VERDICT_NEI


def test_refutation_detected():
    v = aggregate([ev(0.6, 0.02, 0.1, 0.88), ev(0.5, 0.2, 0.7, 0.1)])
    assert v.label == VERDICT_REFUTED and v.decisive_index == 0


def test_irrelevant_evidence_gives_nei():
    v = aggregate([ev(0.10, 0.95, 0.03, 0.02)], relevance_threshold=0.35)
    assert v.label == VERDICT_NEI


def test_conflicting_evidence_flagged():
    v = aggregate([ev(0.6, 0.9, 0.05, 0.05), ev(0.6, 0.05, 0.05, 0.9)])
    assert v.conflicting


def test_empty_evidence():
    v = aggregate([])
    assert v.label == VERDICT_NEI and v.decisive_index is None
