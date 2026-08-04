"""Task 4: chunk standardized Markdown and index it in ChromaDB.

The current project variant uses semantic statistical chunking. The previous
token/character settings are kept as a baseline label for evaluation history.
"""

from pathlib import Path


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BASELINE_CHUNKING_METHOD = "token_text"
CHUNKING_METHOD = "statistical"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"


def load_documents() -> list[dict]:
    """Load non-empty Markdown documents from ``data/standardized``."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if relative_path.parts else "unknown"
        if doc_type not in {"legal", "news"}:
            doc_type = "other"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": relative_path.as_posix(),
                    "filename": md_file.name,
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents with semantic_chunkers.StatisticalChunker."""
    if not documents:
        return []

    chunks: list[dict] = []
    for document in documents:
        content = str(document.get("content", "")).strip()
        if not content:
            continue

        metadata = dict(document.get("metadata", {}))
        source = str(metadata.get("source", "unknown"))
        for chunk_index, chunk_text in enumerate(_split_text(content)):
            if not chunk_text.strip():
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "source": source,
                        "chunk_index": chunk_index,
                    },
                }
            )
    return chunks


def _split_text(content: str) -> list[str]:
    """Split text with StatisticalChunker and cap oversized chunks."""
    try:
        chunk_texts = _statistical_split(content)
    except Exception as exc:
        raise RuntimeError(
            "Statistical chunking requires semantic-router and semantic-chunkers. "
            "Run `pip install -r requirements.txt` before Task 4."
        ) from exc

    chunks: list[str] = []
    for chunk_text in chunk_texts:
        text = str(chunk_text).strip()
        if not text:
            continue
        if len(text) <= int(CHUNK_SIZE * 1.1):
            chunks.append(text)
        else:
            chunks.extend(_simple_recursive_split(text))
    return chunks


def _statistical_split(content: str) -> list[str]:
    """Use the requested semantic-router + semantic-chunkers chunker."""
    from semantic_router.encoders import HuggingFaceEncoder
    from semantic_chunkers import StatisticalChunker

    encoder = HuggingFaceEncoder()
    statistical_chunker = StatisticalChunker(encoder=encoder)

    try:
        raw_chunks = statistical_chunker(docs=[content])
    except TypeError:
        try:
            raw_chunks = statistical_chunker([content])
        except TypeError:
            raw_chunks = statistical_chunker.chunk(content)

    return _coerce_chunk_texts(raw_chunks)


def _coerce_chunk_texts(raw_chunks) -> list[str]:
    """Normalize semantic-chunkers outputs to a list of text strings."""
    texts: list[str] = []

    def visit(value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            texts.append(value)
            return
        if isinstance(value, dict):
            for key in ("content", "text", "page_content"):
                if key in value:
                    visit(value[key])
                    return
            if "splits" in value:
                visit(value["splits"])
            return
        if hasattr(value, "content"):
            visit(value.content)
            return
        if hasattr(value, "text"):
            visit(value.text)
            return
        if hasattr(value, "splits"):
            splits = value.splits
            if isinstance(splits, list):
                joined = " ".join(str(split).strip() for split in splits if str(split).strip())
                if joined:
                    texts.append(joined)
            else:
                visit(splits)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(raw_chunks)
    return [text.strip() for text in texts if text and text.strip()]


def _simple_recursive_split(content: str) -> list[str]:
    """Dependency-free chunker that respects CHUNK_SIZE and overlap."""
    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _split_by_separators(content.strip(), separators)

    chunks: list[str] = []
    for raw_chunk in raw_chunks:
        text = raw_chunk.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(chunk)
            if start + CHUNK_SIZE >= len(text):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _split_by_separators(text: str, separators: list[str]) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    if not separators:
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    separator = separators[0]
    if separator == "":
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    parts = text.split(separator)
    chunks: list[str] = []
    current = ""
    joiner = separator

    for part in parts:
        candidate = f"{current}{joiner}{part}" if current else part
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.extend(_split_by_separators(current, separators[1:]))
        current = part

    if current:
        chunks.extend(_split_by_separators(current, separators[1:]))
    return chunks


def get_embedding_model():
    """Load the embedding model lazily, so importing this module stays cheap."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts using BAAI/bge-m3 as Chroma-compatible float lists."""
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return [embedding.tolist() for embedding in embeddings]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Add an ``embedding`` list to every chunk."""
    if not chunks:
        return []

    embeddings = embed_texts([chunk["content"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def get_collection():
    """Get the persistent cosine-similarity Chroma collection."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict]):
    """Upsert embedded chunks into the persistent Chroma collection."""
    collection = get_collection()
    if not chunks:
        return collection

    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError(
            "Every chunk must contain an 'embedding'. "
            "Call embed_chunks() before index_to_vectorstore()."
        )

    ids = [
        f"{chunk['metadata'].get('source', 'unknown')}_chunk_"
        f"{chunk['metadata'].get('chunk_index', index)}"
        for index, chunk in enumerate(chunks)
    ]
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline():
    """Run load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(
        f"  Chunking: {CHUNKING_METHOD} "
        f"(baseline={BASELINE_CHUNKING_METHOD}, size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    documents = load_documents()
    print(f"\nLoaded {len(documents)} documents")
    if not documents:
        raise FileNotFoundError(
            f"No Markdown documents found in {STANDARDIZED_DIR}. "
            "Complete Task 3 before running Task 4."
        )

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")
    collection = index_to_vectorstore(chunks)
    print(f"Indexed {collection.count()} chunks in '{COLLECTION_NAME}'")


if __name__ == "__main__":
    run_pipeline()
