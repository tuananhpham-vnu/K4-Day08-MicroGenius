"""Task 4: chunk standardized Markdown and index it in ChromaDB.

The project standard is 800-character chunks with 100-character overlap and
the multilingual BAAI/bge-m3 embedding model.
"""

from pathlib import Path


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

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
    """Split documents with a recursive character splitter."""
    if not documents:
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )

    chunks: list[dict] = []
    for document in documents:
        content = str(document.get("content", "")).strip()
        if not content:
            continue

        metadata = dict(document.get("metadata", {}))
        source = str(metadata.get("source", "unknown"))
        for chunk_index, chunk_text in enumerate(splitter.split_text(content)):
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
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
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
