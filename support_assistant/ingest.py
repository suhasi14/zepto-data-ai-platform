
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents():
    """Read all policy documents from the docs folder."""
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "id": file_path.stem,
                "text": text,
                "source": file_path.name
            })

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 policy documents, but found {len(documents)}."
        )

    return documents


def split_into_chunks(text, chunk_size=100, overlap=20):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def create_vector_database():
    """Generate embeddings and store policy chunks in ChromaDB."""
    documents = load_documents()

    print(f"Loaded {len(documents)} policy documents.")

    # Load local sentence-transformer model
    model = SentenceTransformer(MODEL_NAME)

    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Recreate collection to prevent duplicate records
    try:
        client.delete_collection("zepto_policies")
    except Exception:
        pass

    collection = client.create_collection(
        name="zepto_policies",
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []

    # Split each document into chunks
    for document in documents:
        chunks = split_into_chunks(document["text"])

        for index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(
                f'{document["id"]}_chunk_{index}'
            )

            all_metadatas.append({
                "source": document["source"],
                "chunk_number": index
            })

    # Generate embeddings
    embeddings = model.encode(
        all_chunks,
        normalize_embeddings=True
    ).tolist()

    # Store chunks, embeddings and metadata
    collection.add(
        ids=all_ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )

    print(f"Created {len(all_chunks)} text chunks.")
    print(f"Stored embeddings in: {DB_DIR}")
    print("Vector database created successfully!")


if __name__ == "__main__":
    create_vector_database()
