"""Task 9: hybrid retrieval pipeline with vectorless fallback."""

from .task7_reranking import rerank_rrf


# The offline hash-embedding fallback has a lower score range than real BGE.
# Fallback is also gated by lexical evidence below, so a low dense score alone
# does not discard a useful hybrid result.
SCORE_THRESHOLD = 0.20
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Run semantic + lexical search, RRF merge, and PageIndex fallback."""
    if top_k <= 0 or not query.strip():
        return []

    dense_results = _safe_semantic_search(query, top_k=top_k * 2)
    sparse_results = _safe_lexical_search(query, top_k=top_k * 2)

    best_dense_score = float(dense_results[0]["score"]) if dense_results else 0.0
    best_sparse_score = float(sparse_results[0]["score"]) if sparse_results else 0.0
    if best_dense_score < score_threshold and best_sparse_score <= 0.0:
        fallback = _safe_pageindex_search(query, top_k=top_k)
        if fallback:
            return _with_source(fallback[:top_k], "pageindex")

    ranked_lists = [results for results in (dense_results, sparse_results) if results]
    if not ranked_lists:
        fallback = _safe_pageindex_search(query, top_k=top_k)
        return _with_source(fallback[:top_k], "pageindex")

    final_results = (
        rerank_rrf(ranked_lists, top_k=top_k)
        if use_reranking
        else _merge_without_reranking(ranked_lists, top_k=top_k)
    )
    return _with_source(_diversify(final_results, top_k), "hybrid")


def _safe_semantic_search(query: str, top_k: int) -> list[dict]:
    try:
        from .task5_semantic_search import semantic_search

        return semantic_search(query, top_k=top_k)
    except Exception:
        return []


def _safe_lexical_search(query: str, top_k: int) -> list[dict]:
    try:
        from .task6_lexical_search import lexical_search

        return lexical_search(query, top_k=top_k)
    except Exception:
        return []


def _safe_pageindex_search(query: str, top_k: int) -> list[dict]:
    try:
        from .task8_pageindex_vectorless import pageindex_search

        return pageindex_search(query, top_k=top_k)
    except Exception:
        return []


def _merge_without_reranking(ranked_lists: list[list[dict]], top_k: int) -> list[dict]:
    """Stable de-duplicated merge for callers that disable reranking."""
    merged: list[dict] = []
    seen: set[str] = set()

    for ranked_list in ranked_lists:
        for item in ranked_list:
            content = str(item.get("content", ""))
            if not content or content in seen:
                continue
            seen.add(content)
            merged.append(dict(item))
            if len(merged) >= top_k:
                return merged
    return merged


def _with_source(results: list[dict], source: str) -> list[dict]:
    normalized: list[dict] = []
    for item in results:
        result = dict(item)
        if isinstance(item.get("metadata"), dict):
            result["metadata"] = dict(item["metadata"])
        result["score"] = float(result.get("score", 0.0))
        result["source"] = source
        normalized.append(result)
    return normalized


def _diversify(results: list[dict], top_k: int) -> list[dict]:
    """Remove duplicate chunks and cap repeated sources in the context."""
    selected: list[dict] = []
    seen_content: set[str] = set()
    source_counts: dict[str, int] = {}
    for item in results:
        content = str(item.get("content", "")).strip()
        if not content or content in seen_content:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source = str(metadata.get("source") or metadata.get("filename") or "unknown")
        if source_counts.get(source, 0) >= 2:
            continue
        selected.append(item)
        seen_content.add(content)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


if __name__ == "__main__":
    for result in retrieve("return refund policy", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:100]}...")
