"""Task 10: generation with citation-ready context.

The lab can run without an external LLM/API key, so generation here is a
deterministic extractive step over retrieved evidence.
"""

from __future__ import annotations

import re


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = "local-extractive"

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
    """Place strong chunks at the front and end: front + back[::-1]."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


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
    terms = {term.lower() for term in re.findall(r"[\w]+", query, flags=re.UNICODE)}
    return {term for term in terms if len(term) > 2 and term not in STOPWORDS}


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

    def sentence_score(sentence: str) -> tuple[int, int]:
        sentence_terms = {
            term.lower() for term in re.findall(r"[\w]+", sentence, flags=re.UNICODE)
        }
        return (len(query_terms & sentence_terms), -len(sentence))

    return max(sentences, key=sentence_score)


def generate_with_citation(
    query: str,
    context_chunks: list[dict] | None = None,
    top_k: int = TOP_K,
) -> dict:
    """Return an extractive answer with citations and source chunks.

    TOP_P=0.9 and TEMPERATURE=0.3 are documented LLM defaults: broad enough to
    phrase naturally, low enough for factual QA. The local extractor is used so
    tests and demos still work without a networked model.
    """
    if context_chunks is None:
        try:
            from .task9_retrieval_pipeline import retrieve

            chunks = retrieve(query, top_k=top_k)
        except Exception:
            chunks = []
    else:
        chunks = context_chunks
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    if not chunks:
        answer = UNKNOWN_ANSWER
    else:
        query_terms = _query_terms(query)
        answer_parts = []
        used_sources: set[str] = set()

        for chunk in reordered:
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

        answer = " ".join(answer_parts) if answer_parts else UNKNOWN_ANSWER

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "context": context,
    }


if __name__ == "__main__":
    print(generate_with_citation("What payment methods does Shopee support?")["answer"])
