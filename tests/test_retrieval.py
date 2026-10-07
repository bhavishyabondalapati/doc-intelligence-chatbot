import numpy as np
import pytest

from docintel.retrieval import (
    BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever, make_retriever,
    reciprocal_rank_fusion, tokenize,
)

CHUNKS = [
    {"text": "Warfarin dosing should be individualized using the INR."},
    {"text": "Sertraline is a selective serotonin reuptake inhibitor."},
    {"text": "Atorvastatin lowers LDL cholesterol; take 10 mg to 80 mg daily."},
]


class FakeEncoder:
    """Bag-of-letters 'embedding' - deterministic and needs no download."""

    def encode(self, texts, **kwargs):
        vecs = np.zeros((len(texts), 26), dtype="float32")
        for row, text in enumerate(texts):
            for ch in text.lower():
                if "a" <= ch <= "z":
                    vecs[row, ord(ch) - 97] += 1
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class FakeCrossEncoder:
    def predict(self, pairs):
        return [float("serotonin" in text.lower()) for _, text in pairs]


def test_tokenize_keeps_decimals_and_drops_stopwords():
    assert tokenize("What is the 2.5 mg dose of the drug?") == ["2.5", "mg", "dose", "drug"]


def test_bm25_finds_exact_term():
    result = BM25Retriever().index(CHUNKS).search("INR monitoring for warfarin", k=2)
    assert result.hits[0][0] == 0


def test_dense_search_and_k_larger_than_corpus():
    dense = DenseRetriever("fake", encoder=FakeEncoder()).index(CHUNKS)
    result = dense.search("Sertraline serotonin reuptake inhibitor", k=10)
    assert result.hits[0][0] == 1
    assert len(result.hits) == 3  # no -1 padding from FAISS
    assert result.confidence == pytest.approx(result.hits[0][1])


def test_dense_save_and_load_roundtrip(tmp_path):
    dense = DenseRetriever("fake", encoder=FakeEncoder()).index(CHUNKS)
    dense.save(tmp_path, CHUNKS)
    loaded, chunks = DenseRetriever.load(tmp_path, encoder=FakeEncoder())
    assert loaded.model_name == "fake" and chunks == CHUNKS
    assert loaded.search("warfarin INR", 1).hits[0][0] == dense.search("warfarin INR", 1).hits[0][0]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], rrf_k=60)
    assert [idx for idx, _ in fused][:2] == [1, 3]
    assert dict(fused)[1] == pytest.approx(1 / 61 + 1 / 62)


def test_hybrid_uses_dense_cosine_as_confidence():
    dense = DenseRetriever("fake", encoder=FakeEncoder())
    hybrid = HybridRetriever(dense, BM25Retriever()).index(CHUNKS)
    result = hybrid.search("atorvastatin LDL cholesterol", k=3)
    assert result.hits[0][0] == 2
    assert result.confidence == pytest.approx(dense.search("atorvastatin LDL cholesterol", 1).confidence)


def test_reranker_reorders_candidates():
    base = BM25Retriever()
    rerank = RerankRetriever(base, "fake", cross_encoder=FakeCrossEncoder()).index(CHUNKS)
    result = rerank.search("warfarin INR dosing", k=3)
    assert result.hits[0][0] == 1  # the fake cross-encoder prefers the serotonin chunk
    assert result.confidence == 1.0


def test_make_retriever_rejects_index_built_with_other_model():
    dense = DenseRetriever("some-other-model", encoder=FakeEncoder())
    with pytest.raises(ValueError, match="index was built with"):
        make_retriever("dense-bge-small", dense=dense)


def test_make_retriever_shapes():
    assert isinstance(make_retriever("bm25"), BM25Retriever)
    assert isinstance(make_retriever("hybrid-bge-small"), HybridRetriever)
    assert isinstance(make_retriever("hybrid-bge-small+rerank"), RerankRetriever)
