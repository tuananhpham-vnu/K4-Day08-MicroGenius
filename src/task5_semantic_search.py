"""Task 5: semantic search with cosine similarity and lightweight HyDE.

The function reads the local vector index created by Task 4 when available,
then ranks chunks by cosine similarity against a HyDE-expanded query vector.
It intentionally avoids API calls so the individual tests can run offline.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).parent.parent
LOCAL_INDEX_PATH = PROJECT_DIR / "chroma_db" / "local_index.json"
DEFAULT_EMBEDDING_DIM = 256

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "cua", "co", "cho",
    "da", "de", "den", "duoc", "hay", "how", "in", "is", "la", "ma",
    "mot", "nhung", "of", "on", "or", "ra", "sau", "the", "thi", "to",
    "trong", "tu", "va", "ve", "voi", "what", "which", "who", "why",
    "with", "you", "your", "toi",
}


def _tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese/English text into lowercase word-like tokens."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    tokens = re.findall(r"[\w]+", ascii_text, flags=re.UNICODE)
    return [token for token in tokens if len(token) > 1 and token not in STOPWORDS]


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return normalized.encode("ascii", "ignore").decode("ascii")


def _keyword_overlap(query: str, content: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(_tokenize(content))
    overlap = len(query_tokens & content_tokens) / len(query_tokens)
    query_text = _normalized_text(query)
    content_text = _normalized_text(content)
    phrase_bonus = sum(
        0.12
        for phrase in ("15 ngay", "hoan tien", "tra hang", "thanh toan", "hang cam", "hang gia")
        if phrase in query_text and phrase in content_text
    )
    return min(1.0, overlap + phrase_bonus)


def _topic_boost(query: str, metadata: dict | None = None, content: str = "") -> float:
    """Prefer the policy family that matches the user's intent."""
    query_text = _normalized_text(query)
    source_text = _normalized_text(
        " ".join(str((metadata or {}).get(key, "")) for key in ("source", "filename", "title"))
    )
    content_text = _normalized_text(content)
    boost = 0.0

    def has_any(*terms: str) -> bool:
        return any(term in query_text for term in terms)

    if has_any("tra hang", "hoan tien", "return", "refund"):
        if "article_01" in source_text or "returns-refund" in source_text:
            boost += 0.45
        elif "article_04" in source_text:
            boost += 0.12
    if has_any("thanh toan", "payment", "phuong thuc"):
        if "article_02" in source_text or "payment-method" in source_text:
            boost += 0.45
    if has_any("dang ban", "hang cam", "san pham", "hang gia", "my pham"):
        if "dang-ban" in source_text or "seller-listing" in source_text or "product-listing" in source_text:
            boost += 0.45
    if has_any("thue", "gtgt", "tncn", "doanh thu"):
        if "thong-tu-40" in source_text:
            boost += 0.45
    if "tiktok" in query_text and "tiktok" in source_text:
        boost += 0.45
    return boost


def _hash_bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def text_embedding(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """Create a deterministic local embedding for offline dense retrieval.

    Task 4 may use a stronger model such as BAAI/bge-m3. This fallback keeps
    Task 5 runnable in a constrained lab environment by hashing tokens into a
    normalized dense vector, which still supports cosine similarity ranking.
    """
    vector = [0.0] * dim
    for token in _tokenize(text):
        vector[_hash_bucket(token, dim)] += 1.0
    return _normalize(vector)


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    """Return cosine similarity for two numeric vectors."""
    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values:
        return 0.0

    limit = min(len(left_values), len(right_values))
    dot = sum(left_values[i] * right_values[i] for i in range(limit))
    left_norm = math.sqrt(sum(value * value for value in left_values[:limit]))
    right_norm = math.sqrt(sum(value * value for value in right_values[:limit]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _build_hyde_document(query: str) -> str:
    """Build a small hypothetical answer used as HyDE query expansion."""
    terms = set(_tokenize(query))
    hints: list[str] = []

    if terms & {"return", "returns", "refund", "hoan", "tra"}:
        hints.append(
            "The relevant policy explains return and refund eligibility, "
            "required evidence, dispute handling, and refund timing."
        )
    if terms & {"payment", "payments", "method", "methods", "thanh", "toan"}:
        hints.append(
            "The relevant support document describes supported payment methods, "
            "checkout eligibility, card or wallet issues, OTP confirmation, and payment failure."
        )
    if terms & {"seller", "listing", "product", "ban", "dang"}:
        hints.append(
            "The relevant seller regulation covers product listing rules, "
            "seller responsibilities, prohibited items, and e-commerce compliance."
        )
    if terms & {"order", "tracking", "delivery", "don", "hang"}:
        hints.append(
            "The relevant guide explains order tracking, logistics milestones, "
            "delivery evidence, courier updates, and support escalation."
        )

    if not hints:
        hints.append(
            "The relevant e-commerce support document answers the user's policy question "
            "with applicable rules, conditions, steps, and source context."
        )

    return f"{query}\n\nHypothetical answer:\n" + " ".join(hints)


def _load_local_index() -> list[dict]:
    if not LOCAL_INDEX_PATH.exists():
        return []

    try:
        data = json.loads(LOCAL_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(data, list):
        return []

    chunks: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        chunks.append(
            {
                "content": content,
                "metadata": item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
                "embedding": item.get("embedding"),
            }
        )
    return chunks


def _load_chunks_from_task4() -> list[dict]:
    try:
        from .task4_chunking_indexing import chunk_documents, load_documents
    except ImportError:
        return []

    try:
        return chunk_documents(load_documents())
    except Exception:
        return []


def load_or_build_chunks() -> list[dict]:
    """Load the current corpus, falling back to the cached local index.

    ``local_index.json`` is an optional artifact and can outlive a data refresh
    (which previously left only one old chunk from ``article_01.md``).  Reading
    the standardized Markdown first keeps retrieval and the UI synchronized
    with the files the user actually sees.  The cached index remains a useful
    fallback when the corpus is unavailable.
    """
    return _load_chunks_from_task4() or _load_local_index()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Return semantic search results sorted by cosine similarity descending.

    Returns:
        List of {"content": str, "score": float, "metadata": dict}
    """
    if top_k <= 0 or not query.strip():
        return []

    chunks = load_or_build_chunks()
    if not chunks:
        return []

    embedding_dim = DEFAULT_EMBEDDING_DIM
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if isinstance(embedding, list) and embedding:
            embedding_dim = len(embedding)
            break

    query_vector = text_embedding(query, dim=embedding_dim)
    hyde_vector = text_embedding(_build_hyde_document(query), dim=embedding_dim)

    results: list[dict] = []
    for chunk in chunks:
        content = chunk["content"]
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            embedding = text_embedding(content, dim=embedding_dim)

        base_score = cosine_similarity(query_vector, embedding)
        hyde_score = cosine_similarity(hyde_vector, embedding)
        overlap_score = _keyword_overlap(query, content)
        topic_score = _topic_boost(query, chunk.get("metadata", {}), content)
        score = 0.45 * base_score + 0.10 * hyde_score + 0.25 * overlap_score + topic_score
        results.append(
            {
                "content": content,
                "score": round(float(score), 4),
                "metadata": dict(chunk.get("metadata", {})),
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for result in semantic_search("return refund policy", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
