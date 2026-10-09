from pathlib import Path
import os
import hashlib

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

load_dotenv()

# -----------------------------------------
# 1. Knowledge base location
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

def chunk_documents(documents):
    """Split documents into smaller overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:
        document_chunks = splitter.split_text(document["text"])

        for index, chunk in enumerate(document_chunks):
            chunks.append({
                "id": f"{document['source']}_{index}",
                "text": chunk,
                "source": document["source"],
                "chunk_index": index
            })

    return chunks


# -----------------------------------------
# 4. Embedding model
# -----------------------------------------

GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768

_gemini_api_key = os.getenv("GEMINI_API_KEY")

if not _gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

_gemini_client = genai.Client(
    api_key=_gemini_api_key
)


# -----------------------------------------
# 5. Create embeddings
# -----------------------------------------

def create_embeddings(chunks):
    """Create an embedding vector for every chunk."""

    embeddings = []

    for chunk in chunks:
        result = _gemini_client.models.embed_content(
            model=GEMINI_EMBEDDING_MODEL,
            contents=chunk["text"],
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            )
        )

        embeddings.append(result.embeddings[0].values)

    return embeddings


# -----------------------------------------
# 6. ChromaDB
# -----------------------------------------

CHROMA_DB_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR)
)

# Existing Knowledge Base collection.
# Keep this collection intact.

collection = client.get_or_create_collection(
    name="project_management_knowledge"
)

# Separate collection for user-uploaded documents.

uploaded_collection = client.get_or_create_collection(
    name="user_uploaded_documents"
)


# -----------------------------------------
# 7. Store existing Knowledge Base embeddings
# -----------------------------------------

def store_embeddings(chunks, embeddings):
    """Store existing Knowledge Base chunks in ChromaDB."""

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
# 8. Index user-uploaded documents
# -----------------------------------------

def index_uploaded_documents(user_id, documents):
    """
    Index multiple uploaded documents for a specific user.

    Expected input:
    [
        {
            "source": "document.pdf",
            "text": "Extracted document text..."
        }
    ]
    """

    if not user_id:
        raise ValueError("A user_id is required.")

    if not documents:
        return {
            "success": True,
            "files": [],
            "file_count": 0,
            "chunk_count": 0
        }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    indexed_files = []
    total_chunks = 0

    for document in documents:
        source = Path(document["source"]).name
        text = document["text"].strip()

        if not text:
            continue

        # Remove the previous version of this filename
        # for this user before indexing its new contents.

        uploaded_collection.delete(
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"source": source}
                ]
            }
        )

        text_chunks = splitter.split_text(text)

        chunks = []

        for index, chunk_text in enumerate(text_chunks):
            chunk_id = hashlib.sha256(
                (
                    f"{user_id}:{source}:"
                    f"{index}:{chunk_text}"
                ).encode("utf-8")
            ).hexdigest()

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "chunk_index": index
            })

        if not chunks:
            continue

        embeddings = create_embeddings(chunks)

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
# 9. List uploaded documents
# -----------------------------------------

def list_uploaded_documents(user_id):
    """Return the uploaded filenames belonging to one user."""

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
# 10. Remove an uploaded document
# -----------------------------------------

def remove_uploaded_document(user_id, source):
    """Remove one uploaded document belonging to one user."""

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
# 11. Search the existing Knowledge Base
# -----------------------------------------

def search_knowledge_base(query, top_k=3):
    """Search the existing project management Knowledge Base."""

    if collection.count() == 0:
        print("Knowledge Base is empty. Initializing...")

        documents = load_documents()
        chunks = chunk_documents(documents)

        if chunks:
            embeddings = create_embeddings(chunks)
            store_embeddings(chunks, embeddings)

    if collection.count() == 0:
        return []

    result = _gemini_client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    query_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    retrieved_documents = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved_documents.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance
        })

    return retrieved_documents


# -----------------------------------------
# 12. Search uploaded documents
# -----------------------------------------

def search_uploaded_documents(query, user_id, top_k=3):
    """Search only the uploaded documents belonging to one user."""

    if not user_id or uploaded_collection.count() == 0:
        return []

    # Embed the question.

    result = _gemini_client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    query_embedding = result.embeddings[0].values

    # Count only this user's chunks.

    user_results = uploaded_collection.get(
        where={"user_id": str(user_id)},
        include=[]
    )

    user_chunk_count = len(user_results.get("ids", []))

    if user_chunk_count == 0:
        return []

    results = uploaded_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, user_chunk_count),
        where={"user_id": str(user_id)}
    )

    retrieved_documents = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved_documents.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance
        })

    return retrieved_documents


# -----------------------------------------
# 13. Search both document sources
# -----------------------------------------

def search_all_documents(query, user_id=None, top_k=3):
    """
    Search the existing Knowledge Base and, when user_id is
    provided, that user's uploaded documents.

    Results from both sources are combined and ranked by
    ChromaDB distance.
    """

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