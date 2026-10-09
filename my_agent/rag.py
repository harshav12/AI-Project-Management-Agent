from pathlib import Path
import hashlib
from functools import lru_cache

from dotenv import load_dotenv
import chromadb

load_dotenv()

# -----------------------------------------
# 1. Knowledge-base location
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "Knowledge_base"


# -----------------------------------------
# 2. Load knowledge-base documents
# -----------------------------------------

def load_documents():
    """Load all .txt files from the existing Knowledge Base."""

    documents = []

    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        if text.strip():
            documents.append({
                "source": file_path.name,
                "text": text
            })

    return documents


# -----------------------------------------
# 3. Split documents into chunks
# -----------------------------------------

def split_text(text, chunk_size=500, chunk_overlap=100):
    """Split text into overlapping chunks."""

    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            boundary = text.rfind(" ", start, end)

            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(start + 1, end - chunk_overlap)

    return chunks


def chunk_documents(documents):
    """Split knowledge-base documents into chunks."""

    chunks = []

    for document in documents:
        document_chunks = split_text(document["text"])

        for index, chunk in enumerate(document_chunks):
            chunks.append({
                "id": f"{document['source']}_{index}",
                "text": chunk,
                "source": document["source"],
                "chunk_index": index
            })

    return chunks


# -----------------------------------------
# 4. Local embedding model
# -----------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the model only when embeddings are first required."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def create_embeddings(chunks):
    """Generate local embeddings for document chunks."""

    if not chunks:
        return []

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embeddings.tolist()


def create_query_embedding(query):
    """Generate a local embedding for a search query."""

    model = get_embedding_model()

    embedding = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return embedding[0].tolist()


# -----------------------------------------
# 5. ChromaDB
# -----------------------------------------

CHROMA_DB_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)

# New collections for local 384-dimensional embeddings.
# Existing Gemini-based collections remain untouched.

collection = client.get_or_create_collection(
    name="project_management_knowledge_local_minilm",
    metadata={"hnsw:space": "cosine"}
)

uploaded_collection = client.get_or_create_collection(
    name="user_uploaded_documents_local_minilm",
    metadata={"hnsw:space": "cosine"}
)


# -----------------------------------------
# 6. Store knowledge-base embeddings
# -----------------------------------------

def store_embeddings(chunks, embeddings):
    """Store knowledge-base chunks and their embeddings."""

    if not chunks:
        return

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"]
            }
            for chunk in chunks
        ],
        embeddings=embeddings
    )

    print(f"Stored {len(chunks)} Knowledge Base chunks.")


# -----------------------------------------
# 7. Index user-uploaded documents
# -----------------------------------------

def index_uploaded_documents(user_id, documents):
    """Index multiple uploaded documents for a specific user."""

    if not user_id:
        raise ValueError("A user_id is required.")

    if not documents:
        return {
            "success": True,
            "files": [],
            "file_count": 0,
            "chunk_count": 0
        }

    indexed_files = []
    total_chunks = 0

    for document in documents:
        source = Path(document["source"]).name
        text = document["text"].strip()

        if not text:
            continue

        text_chunks = split_text(text)
        chunks = []

        for index, chunk_text in enumerate(text_chunks):
            chunk_id = hashlib.sha256(
                f"{user_id}:{source}:{index}:{chunk_text}".encode("utf-8")
            ).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "chunk_index": index
            })

        if not chunks:
            continue

        # Generate embeddings before removing the previously indexed
        # version, so an embedding error does not delete the old version.
        embeddings = create_embeddings(chunks)

        uploaded_collection.delete(
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"source": source}
                ]
            }
        )

        uploaded_collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[
                {
                    "user_id": str(user_id),
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"]
                }
                for chunk in chunks
            ],
            embeddings=embeddings
        )

        indexed_files.append(source)
        total_chunks += len(chunks)

    return {
        "success": True,
        "files": indexed_files,
        "file_count": len(indexed_files),
        "chunk_count": total_chunks
    }


# -----------------------------------------
# 8. List uploaded documents
# -----------------------------------------

def list_uploaded_documents(user_id):
    """Return uploaded filenames belonging to one user."""

    if not user_id or uploaded_collection.count() == 0:
        return []

    result = uploaded_collection.get(
        where={"user_id": str(user_id)},
        include=["metadatas"]
    )

    filenames = {
        metadata["source"]
        for metadata in result.get("metadatas", [])
        if metadata and metadata.get("source")
    }

    return sorted(filenames)


# -----------------------------------------
# 9. Remove an uploaded document from ChromaDB
# -----------------------------------------

def remove_uploaded_document(user_id, source):
    """Remove indexed chunks belonging to one user's document."""

    if not user_id or not source:
        return 0

    result = uploaded_collection.get(
        where={
            "$and": [
                {"user_id": str(user_id)},
                {"source": Path(source).name}
            ]
        },
        include=["metadatas"]
    )

    ids = result.get("ids", [])

    if ids:
        uploaded_collection.delete(ids=ids)

    return len(ids)


# -----------------------------------------
# 10. Search the existing Knowledge Base
# -----------------------------------------

def search_knowledge_base(query, top_k=3):
    """Search the existing project management Knowledge Base."""

    if top_k <= 0:
        return []

    # Index the knowledge base only when a search first needs it.
    # Existing vectors in this new collection are reused on later searches.

    if collection.count() == 0:
        print("Local Knowledge Base collection is empty. Initializing...")

        documents = load_documents()
        chunks = chunk_documents(documents)

        if chunks:
            embeddings = create_embeddings(chunks)
            store_embeddings(chunks, embeddings)

    count = collection.count()

    if count == 0:
        return []

    # Only the question needs a new embedding for this search.
    query_embedding = create_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_documents.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance
        })

    return retrieved_documents


# -----------------------------------------
# 11. Search uploaded documents
# -----------------------------------------

def search_uploaded_documents(query, user_id, top_k=3):
    """Search only uploaded documents belonging to one user."""

    if not user_id or top_k <= 0 or uploaded_collection.count() == 0:
        return []

    # Check for this user's documents before loading the model.
    user_results = uploaded_collection.get(
        where={"user_id": str(user_id)},
        include=[]
    )

    user_chunk_count = len(user_results.get("ids", []))

    if user_chunk_count == 0:
        return []

    query_embedding = create_query_embedding(query)

    results = uploaded_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, user_chunk_count),
        where={"user_id": str(user_id)},
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved_documents.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance
        })

    return retrieved_documents


# -----------------------------------------
# 12. Search both document sources
# -----------------------------------------

def search_all_documents(query, user_id=None, top_k=3):
    """Search the knowledge base and the current user's uploaded documents."""

    if top_k <= 0:
        return []

    knowledge_results = search_knowledge_base(
        query=query,
        top_k=top_k
    )

    uploaded_results = []

    if user_id:
        uploaded_results = search_uploaded_documents(
            query=query,
            user_id=user_id,
            top_k=top_k
        )

    for result in knowledge_results:
        result["document_scope"] = "knowledge_base"

    for result in uploaded_results:
        result["document_scope"] = "user_upload"

    combined_results = knowledge_results + uploaded_results

    combined_results.sort(
        key=lambda item: item["distance"]
    )

    return combined_results[:top_k]