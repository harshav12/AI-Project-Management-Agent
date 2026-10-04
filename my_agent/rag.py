from pathlib import Path
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# -----------------------------------------
# 1. Knowledge base location
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "Knowledge_base"

# -----------------------------------------
# 2. Load knowledge-base documents
# -----------------------------------------

def load_documents():
    """Load all .txt files from the knowledge_base folder."""
    documents = []

    for file_path in KNOWLEDGE_BASE_DIR.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

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

collection = client.get_or_create_collection(
    name="project_management_knowledge"
)

# -----------------------------------------
# 7. Store embeddings in ChromaDB
# -----------------------------------------

def store_embeddings(chunks, embeddings):
    """Store document chunks and their embeddings in ChromaDB."""

    collection.add(
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

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

# -----------------------------------------
# 8. Search the knowledge base
# -----------------------------------------

def search_knowledge_base(query, top_k=3):
    """Search ChromaDB for chunks relevant to the query."""

    # Initialize knowledge base if empty
    if collection.count() == 0:
        print("Knowledge base is empty. Initializing...")

        documents = load_documents()
        chunks = chunk_documents(documents)
        embeddings = create_embeddings(chunks)
        store_embeddings(chunks, embeddings)

    # Convert query into an embedding
    result = _gemini_client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    query_embedding = result.embeddings[0].values

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved_documents = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        retrieved_documents.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance
        })

    return retrieved_documents