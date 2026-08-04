"""Task 6: lexical search with BM25.

The module builds a local BM25 index lazily from the same chunks used by Task 5.
It prefers ``rank_bm25`` when installed, and falls back to a small compatible
implementation so tests can run offline.
"""

from __future__ import annotations

import math
from collections import Counter

from .task5_semantic_search import (
    _keyword_overlap,
    _normalized_text,
    _topic_boost,
    _tokenize,
    load_or_build_chunks,
)


CORPUS: list[dict] = []
_BM25_INDEX = None


class SimpleBM25:
    """Small BM25Okapi-compatible fallback used when rank_bm25 is unavailable."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(tokens) for tokens in tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / max(self.doc_count, 1)
        self.term_counts = [Counter(tokens) for tokens in tokenized_corpus]
        self.doc_freq = Counter(token for tokens in tokenized_corpus for token in set(tokens))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for idx, counts in enumerate(self.term_counts):
            doc_len = self.doc_lengths[idx]
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if tf == 0:
                    continue

                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
                score += idf * (tf * (self.k1 + 1)) / denom
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index from a list of {'content': str, 'metadata': dict}."""
    tokenized_corpus = [_tokenize(str(doc.get("content", ""))) for doc in corpus]

    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:
        return SimpleBM25(tokenized_corpus)


def _load_corpus() -> list[dict]:
    """Load searchable chunks and normalize their shape."""
    chunks = load_or_build_chunks()
    corpus: list[dict] = []

    for chunk in chunks:
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        metadata = chunk.get("metadata", {})
        corpus.append(
            {
                "content": content,
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            }
        )
    return corpus


def _ensure_index():
    global CORPUS, _BM25_INDEX

    if _BM25_INDEX is not None and CORPUS:
        return _BM25_INDEX

    CORPUS = _load_corpus()
    _BM25_INDEX = build_bm25_index(CORPUS) if CORPUS else None
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search chunks with BM25 and return results sorted by score descending.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    if top_k <= 0 or not query.strip():
        return []

    bm25 = _ensure_index()
    if bm25 is None or not CORPUS:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    normalized_query = _normalized_text(query)
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

    results: list[dict] = []
    for idx in ranked_indices:
        raw_score = float(scores[idx])
        content = CORPUS[idx]["content"]
        metadata = CORPUS[idx].get("metadata", {})
        source_text = " ".join(
            str(metadata.get(key, "")) for key in ("source", "filename", "title")
        )
        overlap = _keyword_overlap(query, f"{source_text}\n{content}")
        score = raw_score + 2.5 * overlap + 5.0 * _topic_boost(query, metadata, content)
        if normalized_query and normalized_query in _normalized_text(content):
            score += 1.0
        if score <= 0:
            continue
        results.append(
            {
                "content": content,
                "score": round(score, 4),
                "metadata": dict(metadata),
            }
        )
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    for result in lexical_search("payment methods", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
