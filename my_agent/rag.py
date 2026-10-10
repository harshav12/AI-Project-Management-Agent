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
VIRTUAL_PAGE_SIZE = 3000


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


def paginate_text(text, page_size=VIRTUAL_PAGE_SIZE):
    """Split unpaginated text into deterministic virtual pages."""
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero.")

    text = str(text or "").strip()
    if not text:
        return []

    pages = []
    start = 0

    while start < len(text):
        end = min(start + page_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        page_text = text[start:end].strip()
        if page_text:
            pages.append(page_text)

        start = end
        while start < len(text) and text[start].isspace():
            start += 1

    return pages


def chunk_documents(documents):
    """Split knowledge-base documents into chunks with virtual page numbers."""

    chunks = []

    for document in documents:
        source = Path(document["source"]).name
        pages = paginate_text(document.get("text", ""))
        chunk_index = 0

        for page_number, page_text in enumerate(pages, start=1):
            for chunk_text in split_text(page_text):
                chunks.append({
                    "id": f"{source}_{chunk_index}",
                    "text": chunk_text,
                    "source": source,
                    "chunk_index": chunk_index,
                    "page": page_number,
                    "page_type": "virtual",
                })
                chunk_index += 1

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
    name="project_management_knowledge_local_minilm_v2_pages",
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
                "chunk_index": chunk["chunk_index"],
                "page": chunk["page"],
                "page_type": chunk["page_type"],
            }
            for chunk in chunks
        ],
        embeddings=embeddings
    )

    print(f"Stored {len(chunks)} Knowledge Base chunks.")


# -----------------------------------------
# 7. Index user-uploaded documents
# -----------------------------------------

def chunk_uploaded_document(user_id, source, document):
    """Split an uploaded document into chunks while preserving or assigning pages."""
    pages = document.get("pages")

    if pages is None:
        pages = [
            {
                "page": page_number,
                "page_type": "virtual",
                "text": page_text,
            }
            for page_number, page_text in enumerate(
                paginate_text(document.get("text", "")),
                start=1
            )
        ]

    chunks = []
    chunk_index = 0

    for page_data in pages:
        page_text = str(page_data.get("text", "")).strip()
        if not page_text:
            continue

        page_number = page_data.get("page")
        page_type = page_data.get(
            "page_type",
            document.get("page_type", "physical")
        )

        for chunk_text in split_text(page_text):
            chunk_id = hashlib.sha256(
                (
                    f"{user_id}:{source}:{page_number}:"
                    f"{page_type}:{chunk_index}:{chunk_text}"
                ).encode("utf-8")
            ).hexdigest()

            chunk = {
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "chunk_index": chunk_index,
                "page": int(page_number),
                "page_type": page_type,
            }

            chunks.append(chunk)
            chunk_index += 1

    return chunks


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
        chunks = chunk_uploaded_document(
            user_id=user_id,
            source=source,
            document=document,
        )

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

        metadatas = []
        for chunk in chunks:
            metadata = {
                "user_id": str(user_id),
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "page": chunk["page"],
                "page_type": chunk["page_type"],
            }
            metadatas.append(metadata)

        uploaded_collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=metadatas,
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
            "distance": distance,
            "page": metadata["page"],
            "page_type": metadata["page_type"],
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
        retrieved_document = {
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance
        }
        if metadata.get("page") is not None:
            retrieved_document["page"] = metadata["page"]
            retrieved_document["page_type"] = metadata.get(
                "page_type",
                "physical"
            )
        retrieved_documents.append(retrieved_document)

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