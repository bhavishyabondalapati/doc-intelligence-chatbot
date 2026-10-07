"""
Retrieval: given a question, find the most relevant chunks.

Four strategies, all with the same interface (`index(chunks)` once, then
`search(query, k)`), so the chatbot and the evaluation can swap them:

  DenseRetriever   - embed text with a sentence-transformer, search by
                     cosine similarity in FAISS. Good at meaning
                     ("blood thinner" ~ "anticoagulant"), weaker at exact
                     terms like drug names, numbers or section titles.

  BM25Retriever    - classic keyword search (the algorithm behind most
                     search engines before neural nets). Great at exact
                     terms, blind to synonyms.

  HybridRetriever  - runs both and merges the two ranked lists with
                     Reciprocal Rank Fusion (RRF): each chunk scores
                     sum(1 / (60 + rank)) across lists, so a chunk ranked
                     well by either method rises. RRF only uses ranks, so
                     we never have to compare a cosine to a BM25 score.

  RerankRetriever  - takes the top ~30 candidates from another retriever
                     and re-scores each (question, chunk) pair with a
                     cross-encoder: a model that reads both texts together,
                     which is slower but much more precise than comparing
                     two separately computed vectors.

Every search returns a RetrievalResult with a `confidence` number. The
chatbot abstains ("I don't have enough information") when confidence is
below a threshold; the evaluation picks that threshold from data.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

# BGE embedding models were trained to expect this instruction in front
# of search queries (not in front of the documents).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class RetrievalResult:
    hits: list = field(default_factory=list)  # [(chunk_index, score)], best first
    confidence: float = 0.0                    # how sure we are anything relevant was found


def default_query_prefix(model_name):
    return BGE_QUERY_PREFIX if "bge-" in model_name.lower() else ""


class DenseRetriever:
    name = "dense"

    def __init__(self, model_name, device=None, query_prefix=None, encoder=None, batch_size=32):
        self.model_name = model_name
        self.query_prefix = default_query_prefix(model_name) if query_prefix is None else query_prefix
        self.batch_size = batch_size
        self._encoder = encoder  # tests pass a tiny fake encoder here
        self._device = device
        self.faiss_index = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            from docintel.config import best_device
            self._encoder = SentenceTransformer(self.model_name, device=self._device or best_device())
        return self._encoder

    def _embed(self, texts):
        vectors = self.encoder.encode(
            list(texts), normalize_embeddings=True, batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
        )
        return np.asarray(vectors, dtype="float32")

    def index(self, chunks):
        embeddings = self._embed([c["text"] for c in chunks])
        # Inner product on normalized vectors = cosine similarity
        self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self.faiss_index.add(embeddings)
        return self

    def search(self, query, k=5):
        query_vector = self._embed([self.query_prefix + query])
        k = min(k, self.faiss_index.ntotal)  # FAISS pads with -1 if k > number of chunks
        scores, indices = self.faiss_index.search(query_vector, k)
        hits = [(int(i), float(s)) for i, s in zip(indices[0], scores[0])]
        return RetrievalResult(hits=hits, confidence=hits[0][1] if hits else 0.0)

    def save(self, index_dir: Path, chunks):
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        (index_dir / "metadata.json").write_text(json.dumps(chunks, indent=2))
        # Record which model built the index, so we can refuse to query it
        # with a different one (the vectors would be meaningless).
        (index_dir / "manifest.json").write_text(json.dumps({
            "embed_model": self.model_name, "num_chunks": len(chunks),
        }, indent=2))

    @classmethod
    def load(cls, index_dir: Path, **kwargs):
        manifest = json.loads((index_dir / "manifest.json").read_text())
        retriever = cls(manifest["embed_model"], **kwargs)
        retriever.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        chunks = json.loads((index_dir / "metadata.json").read_text())
        return retriever, chunks


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its of on or "
    "should that the their there this to was what when where which who why will with".split()
)


def tokenize(text):
    """Lowercase word tokens (keeps decimals like '2.5' together), minus stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def index(self, chunks):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        return self

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        hits = [(int(i), float(scores[i])) for i in top]
        return RetrievalResult(hits=hits, confidence=hits[0][1] if hits else 0.0)


def reciprocal_rank_fusion(ranked_lists, rrf_k=60):
    """Merge ranked lists of chunk indices: score = sum over lists of 1 / (rrf_k + rank)."""
    fused = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])


class HybridRetriever:
    name = "hybrid"

    def __init__(self, dense, sparse, candidates=50, rrf_k=60):
        self.dense, self.sparse = dense, sparse
        self.candidates, self.rrf_k = candidates, rrf_k

    def index(self, chunks):
        if self.dense.faiss_index is None:  # skip if loaded from disk already
            self.dense.index(chunks)
        self.sparse.index(chunks)
        return self

    def search(self, query, k=5):
        dense_result = self.dense.search(query, self.candidates)
        sparse_result = self.sparse.search(query, self.candidates)
        fused = reciprocal_rank_fusion(
            [[i for i, _ in dense_result.hits], [i for i, _ in sparse_result.hits]], self.rrf_k
        )
        # RRF scores only describe rank agreement, so they say little about
        # whether the answer exists at all. Use the dense cosine instead.
        return RetrievalResult(hits=fused[:k], confidence=dense_result.confidence)


class RerankRetriever:
    name = "rerank"

    def __init__(self, base, model_name, candidates=30, device=None, cross_encoder=None):
        self.base, self.model_name, self.candidates = base, model_name, candidates
        self._device, self._cross_encoder = device, cross_encoder
        self.chunks = None

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            from docintel.config import best_device
            self._cross_encoder = CrossEncoder(self.model_name, device=self._device or best_device())
        return self._cross_encoder

    def index(self, chunks):
        self.chunks = chunks
        self.base.index(chunks)
        return self

    def search(self, query, k=5):
        candidates = [i for i, _ in self.base.search(query, self.candidates).hits]
        scores = self.cross_encoder.predict([(query, self.chunks[i]["text"]) for i in candidates])
        ranked = sorted(zip(candidates, map(float, scores)), key=lambda item: -item[1])
        return RetrievalResult(hits=ranked[:k], confidence=ranked[0][1] if ranked else 0.0)


# Named retrieval setups compared in eval/run_eval.py. The chatbot uses
# DEFAULT_CONFIG, chosen from those results (see README).
EMBED_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
}
RERANKER_MODEL = "BAAI/bge-reranker-base"

RETRIEVER_CONFIGS = {
    # name: (embedding model key or None, use BM25?, use reranker?)
    "baseline-minilm": ("minilm", False, False),
    "bm25": (None, True, False),
    "dense-bge-small": ("bge-small", False, False),
    "dense-bge-base": ("bge-base", False, False),
    "hybrid-bge-small": ("bge-small", True, False),
    "hybrid-bge-small+rerank": ("bge-small", True, True),
    "hybrid-bge-base+rerank": ("bge-base", True, True),
}
DEFAULT_CONFIG = "hybrid-bge-small"


def embed_model_for(config):
    key = RETRIEVER_CONFIGS[config][0]
    return EMBED_MODELS[key] if key else None


def make_retriever(config, dense=None, device=None):
    """Build an (un-indexed) retriever for a named config. Pass `dense` to reuse a loaded index."""
    embed_key, use_bm25, use_rerank = RETRIEVER_CONFIGS[config]
    if embed_key and dense is None:
        dense = DenseRetriever(EMBED_MODELS[embed_key], device=device)
    if embed_key and dense.model_name != EMBED_MODELS[embed_key]:
        raise ValueError(f"Config {config!r} needs {EMBED_MODELS[embed_key]}, "
                         f"but the index was built with {dense.model_name}.")

    if embed_key and use_bm25:
        retriever = HybridRetriever(dense, BM25Retriever())
    elif use_bm25:
        retriever = BM25Retriever()
    else:
        retriever = dense
    if use_rerank:
        retriever = RerankRetriever(retriever, RERANKER_MODEL, device=device)
    return retriever
