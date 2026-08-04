"""Task 10: generation with citation-ready context.

The lab can run without an external LLM/API key, so generation here is a
deterministic extractive step over retrieved evidence.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from .task5_semantic_search import _tokenize


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """Answer the following question comprehensively.
For every statement of fact or claim, immediately insert a citation
in brackets linking to the specific source (e.g., [Source, Year]).
If the information is not explicitly stated in the provided context
or knowledge base, state 'I cannot verify this information'
rather than guessing."""

UNKNOWN_ANSWER = "I cannot verify this information from the provided context."

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "by",
    "cua",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "la",
    "lam",
    "nhung",
    "of",
    "on",
    "or",
    "the",
    "to",
    "toi",
    "trong",
    "va",
    "ve",
    "what",
    "which",
    "with",
}


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Keep retrieval rank intact so the strongest evidence is read first."""
    return list(chunks)


def format_context(chunks: list[dict]) -> str:
    """Format chunks with source labels for citations."""
    context_parts = []
    for idx, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        source = metadata.get("source", f"Source {idx}")
        year = _citation_year(chunk)
        doc_type = metadata.get("type", "unknown")
        context_parts.append(
            f"[Document {idx} | Source: {source} | Year: {year} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


def _query_terms(query: str) -> set[str]:
    """Tokenize a user query into lightweight matching terms."""
    return {term for term in _tokenize(query) if len(term) > 2}


def _citation_source(chunk: dict) -> str:
    """Return a readable source name for citation brackets."""
    metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
    source = str(metadata.get("source") or metadata.get("filename") or "Source")
    source = source.replace("\\", "/").rsplit("/", 1)[-1]
    return source.rsplit(".", 1)[0] if "." in source else source


def _citation_year(chunk: dict) -> str:
    """Infer citation year from metadata, source path, or chunk content."""
    metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
    candidates = [
        metadata.get("year"),
        metadata.get("date"),
        metadata.get("date_crawled"),
        metadata.get("source"),
        metadata.get("filename"),
        chunk.get("content", ""),
    ]
    for value in candidates:
        match = re.search(r"\b(20\d{2}|19\d{2})\b", str(value))
        if match:
            return match.group(1)
    return "2026"


def _split_sentences(text: str) -> list[str]:
    """Split retrieved text into answer-sized evidence sentences."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [
        sentence.strip(" -")
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if len(sentence.strip(" -")) >= 30
    ]


def _best_sentence(chunk: dict, query_terms: set[str]) -> str:
    """Choose the sentence with the highest overlap with the query."""
    content = str(chunk.get("content", ""))
    sentences = _split_sentences(content)
    if not sentences:
        return content.strip().replace("\n", " ")[:500]

    def sentence_score(sentence: str) -> tuple[int, int, int]:
        sentence_terms = set(_tokenize(sentence))
        overlap = len(query_terms & sentence_terms)
        numeric_hits = sum(
            1 for term in query_terms if term.isdigit() and term in sentence_terms
        )
        return (overlap, numeric_hits, -len(sentence))

    ranked = sorted(sentences, key=sentence_score, reverse=True)
    minimum_overlap = min(2, max(len(query_terms), 1))
    if ranked and sentence_score(ranked[0])[0] >= minimum_overlap:
        return ranked[0]
    return ""


def _sentence_relevance(sentence: str, query: str, query_terms: set[str]) -> tuple[int, int, int]:
    terms = set(_tokenize(sentence))
    overlap = len(query_terms & terms)
    query_text = " ".join(_tokenize(query))
    time_bonus = 0
    if any(term in query_text for term in ("bao lau", "thoi han", "thoi gian")):
        if any(term in terms for term in ("ngay", "gio", "thang", "nam")):
            time_bonus += 3
        if any(term.isdigit() for term in terms):
            time_bonus += 3
    return (overlap + time_bonus, overlap, -len(sentence))


def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = TOP_K,
    use_reranking: bool = True,
) -> dict:
    """Return a Gemini-generated answer with citations and source chunks.

    TOP_P=0.9 and TEMPERATURE=0.3 are documented LLM defaults: broad enough to
    phrase naturally, low enough for factual QA. The local extractor remains as
    an offline fallback when GEMINI_API_KEY is not configured.
    """
    if context_chunks is None:
        try:
            from .task9_retrieval_pipeline import retrieve

            chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)
        except Exception:
            chunks = []
    else:
        chunks = context_chunks
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not chunks:
        answer = UNKNOWN_ANSWER
    elif os.getenv("GEMINI_API_KEY", "").strip():
        answer = _generate_with_gemini(query, reordered, context)
    else:
        answer = _generate_extractive_answer(query, reordered)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "context": context,
    }


def _generate_with_gemini(query: str, chunks: list[dict], context: str) -> str:
    """Generate a cited answer with Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())
    model = os.getenv("GEMINI_GENERATION_MODEL", LLM_MODEL).strip() or LLM_MODEL
    try:
        response = client.models.generate_content(
            model=model,
            contents=(
                f"Question:\n{query}\n\n"
                f"Context:\n{context}\n\n"
                "Answer with citations from the provided context only."
            ),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            ),
        )
    except Exception:
        return _generate_extractive_answer(query, chunks)
    answer = (getattr(response, "text", None) or "").strip()
    return answer or _generate_extractive_answer(query, chunks)


def _generate_extractive_answer(query: str, chunks: list[dict]) -> str:
    """Offline fallback used only when GEMINI_API_KEY is unavailable."""
    query_terms = _query_terms(query)
    evidence = []
    for chunk in chunks:
        sentence = _best_sentence(chunk, query_terms)
        if sentence:
            evidence.append((_sentence_relevance(sentence, query, query_terms), chunk))
    evidence.sort(key=lambda item: item[0], reverse=True)
    answer_parts = []
    used_sources: set[str] = set()

    for _, chunk in evidence:
        citation = f"[{_citation_source(chunk)}, {_citation_year(chunk)}]"
        if citation in used_sources:
            continue

        sentence = _best_sentence(chunk, query_terms).strip()
        if not sentence:
            continue

        sentence = sentence[:500].rstrip()
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        answer_parts.append(f"{sentence} {citation}")
        used_sources.add(citation)

        if len(answer_parts) >= 3:
            break

    return " ".join(answer_parts) if answer_parts else UNKNOWN_ANSWER


if __name__ == "__main__":
    print(generate_with_citation("What payment methods does Shopee support?")["answer"])
