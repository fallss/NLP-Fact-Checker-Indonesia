from fact_checker.config import NLI_LABELS
from fact_checker.evaluation import classification_metrics, load_benchmark


def test_metrics_use_explicit_label_order():
    # Regresi bug versi lama: baris classification_report tertukar
    y_true = ["ENTAILMENT", "ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
    y_pred = ["ENTAILMENT", "ENTAILMENT", "ENTAILMENT", "NEUTRAL"]
    m = classification_metrics(y_true, y_pred, NLI_LABELS)
    assert m["per_class"]["ENTAILMENT"]["recall"] == 1.0
    assert m["per_class"]["CONTRADICTION"]["recall"] == 0.0
    assert m["confusion_matrix"][2] == [1, 0, 0]  # baris CONTRADICTION -> diprediksi ENTAILMENT
    assert m["accuracy"] == 0.75


def test_benchmark_is_balanced_and_grounded():
    cases = load_benchmark()
    with_evidence = [c for c in cases if c.evidence]
    counts = {lbl: sum(c.label == lbl for c in with_evidence) for lbl in NLI_LABELS}
    assert len(set(counts.values())) == 1, counts
    assert all(c.article_id in c.gold_article_ids for c in with_evidence)
