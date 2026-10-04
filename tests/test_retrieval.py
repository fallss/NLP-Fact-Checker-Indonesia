import numpy as np

from fact_checker.retrieval import BM25Index, _top_indices, reciprocal_rank_fusion, tokenize

DOCS = [
    "Kebakaran depo Pertamina Plumpang menewaskan belasan warga",
    "KPU mengajukan banding atas putusan penundaan pemilu",
    "Gunung Merapi mengeluarkan awan panas guguran",
]


def test_tokenize_removes_stopwords():
    assert tokenize("Yang terjadi di Plumpang dan Jakarta") == ["terjadi", "plumpang", "jakarta"]


def test_bm25_ranks_matching_document_first():
    bm25 = BM25Index().fit(DOCS)
    assert int(np.argmax(bm25.score("banding KPU pemilu"))) == 1
    assert int(np.argmax(bm25.score("awan panas Merapi"))) == 2
    assert bm25.score("kata-tidak-ada-di-korpus").sum() == 0


def test_rrf_rewards_consensus():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 3]], k=60)
    assert fused[1] == fused[2] > fused[3]


def test_top_indices_sorted():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert list(_top_indices(scores, 3)) == [1, 3, 2]
