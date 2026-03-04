"""
faiss-backed vector memory system with sentence-transformer embeddings.

this module provides an embedding-based memory system that enables
paper-faithful evaluation of attack success rates (asr-r, asr-a, asr-t)
using real semantic similarity retrieval instead of trivial string matching.

the system uses all-minilm-l6-v2 (384-dim) sentence embeddings with
cosine similarity computed via normalized inner product on a faiss index.

references:
- reimers & gurevych. sentence-bert: sentence embeddings using siamese
  bert-networks. emnlp 2019.
- johnson, douze & jégou. billion-scale similarity search with gpus.
  ieee transactions on big data, 2019.

all comments are lowercase.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from utils.logging import logger

# ---------------------------------------------------------------------------
# module-level lazy import cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: Optional[Any] = None  # cached SentenceTransformer instance
_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_model() -> Any:
    """lazy-load and cache the sentence-transformer model."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        try:
            from sentence_transformers import SentenceTransformer

            logger.logger.info(f"loading sentence-transformer: {_MODEL_NAME}")
            _MODEL_CACHE = SentenceTransformer(_MODEL_NAME)
            logger.logger.info("sentence-transformer loaded successfully")
        except ImportError as exc:
            raise ImportError(
                "sentence_transformers is required for VectorMemorySystem. "
                "install with: pip install sentence-transformers"
            ) from exc
    return _MODEL_CACHE


def _load_faiss() -> Any:
    """lazy-load faiss."""
    try:
        import faiss

        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss is required for VectorMemorySystem. "
            "install with: pip install faiss-cpu"
        ) from exc


# ---------------------------------------------------------------------------
# VectorMemorySystem
# ---------------------------------------------------------------------------


class VectorMemorySystem:
    """
    faiss-backed semantic memory with sentence-transformer retrieval.

    implements the memorysystem protocol using cosine similarity over dense
    embeddings from all-minilm-l6-v2.  designed for paper-faithful asr
    evaluation: poison entries must appear in top-k retrieval for victim
    queries purely based on semantic similarity.

    key properties:
    - 384-dimensional dense embeddings (all-minilm-l6-v2)
    - cosine similarity via l2-normalized inner product on IndexFlatIP
    - efficient batch ingestion for large corpora
    - poison_retrieval_test() measures realistic asr-r

    usage:
        mem = VectorMemorySystem()
        mem.store("key1", "normal task history entry")
        results = mem.search("what tasks do i have today?", top_k=5)
        test = mem.poison_retrieval_test("today's tasks", ["poison_001"])
    """

    EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dimensionality

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        initialize the vector memory system.

        args:
            config: optional configuration dict.  currently unused but
                    preserved for interface compatibility with wrappers.
        """
        self.config = config or {}
        self.logger = logger

        # aligned storage (index position → entry data)
        self._keys: List[str] = []
        self._contents: List[str] = []
        self._metadata: List[Optional[Dict[str, Any]]] = []

        # key → list[int]: all faiss index positions for this key
        self._key_to_positions: Dict[str, List[int]] = {}

        # lazy-initialized faiss index
        self._index: Optional[Any] = None

    # -----------------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------------

    def _get_index(self) -> Any:
        """return (or create) the faiss IndexFlatIP."""
        if self._index is None:
            faiss = _load_faiss()
            self._index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        return self._index

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        embed a list of texts and l2-normalise for cosine similarity.

        args:
            texts: list of strings to embed

        returns:
            float32 array of shape (len(texts), EMBEDDING_DIM)
        """
        model = _load_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return np.array(embeddings, dtype=np.float32)

    def _content_to_str(self, value: Any) -> str:
        """convert any value to a flat string for embedding."""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                parts.append(f"{k}: {v}")
            return " | ".join(parts)
        if isinstance(value, (list, tuple)):
            return " ".join(str(x) for x in value)
        return str(value)

    # -----------------------------------------------------------------------
    # MemorySystem protocol
    # -----------------------------------------------------------------------

    def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        embed and store a key-value pair.

        if the key already exists the new entry is appended alongside the
        old one.  retrieve() returns the most recently stored value; search()
        deduplicates by key returning the highest-scoring match.

        args:
            key: unique identifier for this entry
            value: content to embed and store (any type → converted to str)
            metadata: optional metadata dict attached to this entry
        """
        content_str = self._content_to_str(value)
        embedding = self._embed([content_str])  # shape (1, 384)

        position = len(self._keys)
        self._keys.append(key)
        self._contents.append(content_str)
        self._metadata.append(metadata)

        if key not in self._key_to_positions:
            self._key_to_positions[key] = []
        self._key_to_positions[key].append(position)

        self._get_index().add(embedding)

    def add_batch(self, entries: List[Dict[str, Any]]) -> None:
        """
        efficiently add multiple entries in one embedding pass.

        args:
            entries: list of dicts, each with keys:
                     "key"      (str, required)
                     "content"  (any, required)
                     "metadata" (dict, optional)
        """
        if not entries:
            return

        start_position = len(self._keys)
        contents_str = [self._content_to_str(e.get("content", "")) for e in entries]
        keys = [
            e.get("key", f"entry_{start_position + i}") for i, e in enumerate(entries)
        ]
        metadatas = [e.get("metadata") for e in entries]

        embeddings = self._embed(contents_str)  # (n, 384)

        for i, (key, content, meta) in enumerate(zip(keys, contents_str, metadatas)):
            position = start_position + i
            self._keys.append(key)
            self._contents.append(content)
            self._metadata.append(meta)
            if key not in self._key_to_positions:
                self._key_to_positions[key] = []
            self._key_to_positions[key].append(position)

        self._get_index().add(embeddings)

    def retrieve(self, key: str) -> Optional[str]:
        """
        retrieve the most recently stored content for a given key.

        args:
            key: memory key

        returns:
            content string, or None if key not found
        """
        positions = self._key_to_positions.get(key)
        if not positions:
            return None
        return self._contents[positions[-1]]

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        retrieve top-k semantically similar entries.

        results are deduplicated by key: when multiple positions share the
        same key only the highest-scoring one is returned.

        args:
            query: natural-language search query
            top_k: number of results to return

        returns:
            list of result dicts with keys: key, content, metadata, score, rank
        """
        n = len(self._keys)
        if n == 0:
            return []

        k_search = min(top_k * 3, n)  # over-fetch to allow dedup
        query_emb = self._embed([query])  # (1, 384)
        scores, indices = self._get_index().search(query_emb, k_search)

        results: List[Dict[str, Any]] = []
        seen_keys: set = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= n:
                continue
            key = self._keys[idx]
            if key in seen_keys:
                continue  # deduplicate — keep first (highest score) occurrence
            seen_keys.add(key)
            results.append(
                {
                    "key": key,
                    "content": self._contents[idx],
                    "metadata": self._metadata[idx],
                    "score": float(score),
                    "rank": len(results) + 1,
                }
            )
            if len(results) >= top_k:
                break

        return results

    def poison_retrieval_test(
        self,
        query: str,
        poison_keys: List[str],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        test whether adversarial entries appear in top-k for a victim query.

        this is the core measurement for asr-r (attack success rate —
        retrieval): the proportion of victim queries for which at least one
        adversarial passage appears in the top-k retrieved results.

        args:
            query: victim query to test
            poison_keys: list of keys identifying adversarial entries
            top_k: retrieval depth to evaluate

        returns:
            dict with:
              retrieved_any_poison: bool — whether any poison in top-k
              n_poison_retrieved: int — count of poison keys in top-k
              poison_keys_retrieved: list[str] — which poison keys appeared
              poison_ranks: dict[str, int] — rank of each retrieved poison key
              poison_scores: dict[str, float] — cosine score for each poison
              top_k_results: list[dict] — full top-k result list
        """
        results = self.search(query, top_k=top_k)
        retrieved_set = {r["key"] for r in results}
        poison_set = set(poison_keys)

        retrieved_poison = retrieved_set & poison_set

        poison_ranks: Dict[str, int] = {}
        poison_scores: Dict[str, float] = {}
        for r in results:
            if r["key"] in poison_set:
                poison_ranks[r["key"]] = r["rank"]
                poison_scores[r["key"]] = r["score"]

        return {
            "retrieved_any_poison": len(retrieved_poison) > 0,
            "n_poison_retrieved": len(retrieved_poison),
            "poison_keys_retrieved": list(retrieved_poison),
            "poison_ranks": poison_ranks,
            "poison_scores": poison_scores,
            "top_k_results": results,
            "n_total_retrieved": len(results),
        }

    # -----------------------------------------------------------------------
    # utility
    # -----------------------------------------------------------------------

    def get_all_keys(self) -> List[str]:
        """return all unique keys currently stored."""
        return list(self._key_to_positions.keys())

    def get_size(self) -> int:
        """return total number of stored entries (including duplicates)."""
        return len(self._keys)

    def clear(self) -> None:
        """remove all stored entries and reset the faiss index."""
        self._index = None
        self._keys = []
        self._contents = []
        self._metadata = []
        self._key_to_positions = {}
        self.logger.logger.debug("vector memory system cleared")

    def get_stats(self) -> Dict[str, Any]:
        """return summary statistics about the memory store."""
        return {
            "total_entries": len(self._keys),
            "unique_keys": len(self._key_to_positions),
            "embedding_dim": self.EMBEDDING_DIM,
            "model_name": _MODEL_NAME,
            "index_type": "IndexFlatIP",
        }
