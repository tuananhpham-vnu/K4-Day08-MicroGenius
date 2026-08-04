"""Task 7: reranking utilities.

RRF (Reciprocal Rank Fusion) is the selected implementation for this project.
It combines ranked results from dense and sparse retrieval without requiring
another model or API key.
"""


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Placeholder for an optional cross-encoder implementation."""
    raise NotImplementedError(
        "Cross-encoder reranking is optional; use method='rrf' for this project."
    )


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Placeholder for the optional MMR implementation."""
    raise NotImplementedError(
        "MMR reranking is optional; use rerank_rrf() for this project."
    )


def _candidate_key(candidate: dict) -> tuple:
    """Return a stable identity key for a retrieval result.

    Content is the primary identity because dense and BM25 results normally
    carry the same chunk text but may have different scores. If content is
    absent, source/chunk metadata is used as a fallback.
    """
    if "content" in candidate:
        return ("content", str(candidate["content"]))

    metadata = candidate.get("metadata") or {}
    return (
        "metadata",
        str(metadata.get("source", "")),
        str(metadata.get("chunk_index", "")),
    )


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Fuse ranked result lists using Reciprocal Rank Fusion.

    For every ranker, the first result has rank 1. A document's fused score is:

        RRF(document) = sum(1 / (k + rank))

    The original retrieval score is deliberately ignored. This is important
    because BM25 and cosine similarity use different score scales. The returned
    ``score`` is the RRF score, while the candidate content and metadata are
    preserved.

    Duplicate occurrences in one ranked list count only once, using the first
    (best) rank. Duplicate documents across different rankers receive a score
    contribution from each ranker.
    """
    if top_k <= 0 or not ranked_lists:
        return []
    if k < 0:
        raise ValueError("RRF parameter k must be >= 0")

    scores: dict[tuple, float] = {}
    candidates_by_key: dict[tuple, dict] = {}
    first_seen: dict[tuple, tuple[int, int, str]] = {}

    for list_index, ranked_list in enumerate(ranked_lists):
        if not ranked_list:
            continue

        seen_in_ranker: set[tuple] = set()
        effective_rank = 0

        for candidate in ranked_list:
            if not isinstance(candidate, dict):
                continue

            key = _candidate_key(candidate)
            if key in seen_in_ranker:
                continue
            seen_in_ranker.add(key)
            effective_rank += 1

            contribution = 1.0 / (k + effective_rank)
            scores[key] = scores.get(key, 0.0) + contribution

            # Keep the first complete result, usually the dense result. This
            # avoids replacing useful metadata with a poorer later copy.
            if key not in candidates_by_key:
                candidates_by_key[key] = dict(candidate)
                if isinstance(candidate.get("metadata"), dict):
                    candidates_by_key[key]["metadata"] = dict(candidate["metadata"])
                first_seen[key] = (
                    list_index,
                    effective_rank,
                    str(key),
                )

    ordered_keys = sorted(
        scores,
        key=lambda key: (
            -scores[key],
            first_seen[key][0],
            first_seen[key][1],
            first_seen[key][2],
        ),
    )

    results: list[dict] = []
    for key in ordered_keys[:top_k]:
        result = dict(candidates_by_key[key])
        result["score"] = scores[key]
        result["rrf_score"] = scores[key]
        results.append(result)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """Unified reranking interface.

    ``rerank()`` receives one already-ranked list, so the default RRF mode
    treats it as one ranker. Task 9 should call ``rerank_rrf`` directly with
    both dense and sparse lists to get true hybrid fusion.
    """
    del query  # Reserved for optional cross-encoder/MMR implementations.
    method = method.lower().strip()

    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    if method == "cross_encoder":
        raise NotImplementedError(
            "Cross-encoder reranking is optional; use method='rrf'."
        )
    if method == "mmr":
        raise NotImplementedError(
            "MMR reranking requires query embeddings; use rerank_rrf()."
        )
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dense_results = [
        {"content": "Payment methods", "score": 0.90, "metadata": {"source": "a.md"}},
        {"content": "Return policy", "score": 0.80, "metadata": {"source": "b.md"}},
    ]
    sparse_results = [
        {"content": "Return policy", "score": 4.20, "metadata": {"source": "b.md"}},
        {"content": "Payment methods", "score": 2.10, "metadata": {"source": "a.md"}},
    ]
    for item in rerank_rrf([dense_results, sparse_results], top_k=2):
        print(f"[{item['score']:.4f}] {item['content']}")
