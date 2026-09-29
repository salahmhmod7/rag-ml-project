import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Configuration (plain os.getenv – no pydantic settings conflicts)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-api")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., description="User's question to answer")


class SourceDoc(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDoc] = []


# ---------------------------------------------------------------------------
# Global state (initialized once at startup)
# ---------------------------------------------------------------------------

chroma_client = None
chroma_collection = None
embedding_model = None
reranker = None
ollama_available = False


# ---------------------------------------------------------------------------
# Lifespan – initialize heavy resources once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_client, chroma_collection, embedding_model, reranker, ollama_available

    logger.info("Starting up RAG API - loading models and vector store")

    # ChromaDB persistent client
    persist_path = Path(CHROMA_PERSIST_DIRECTORY)
    persist_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(persist_path))

    # Get or create a collection for documents
    chroma_collection = chroma_client.get_or_create_collection(
        name="documents",
        metadata={"hnsw_space": "cosine"},
    )
    logger.info(f"ChromaDB ready (collection: {chroma_collection.name})")

    # SentenceTransformer for embeddings (BAAI/bge-m3, CPU mode)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    logger.info(f"Embedding model loaded: {EMBEDDING_MODEL_NAME}")

    # CrossEncoder reranker (CPU mode)
    reranker = CrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=DEVICE,
    )
    logger.info("Reranker model loaded")

    # Test Ollama connectivity
    global ollama_available
    try:
        import ollama
        ollama.list()
        ollama_available = True
        logger.info("Ollama is reachable")
    except Exception as e:
        logger.warning(f"Ollama not reachable at {OLLAMA_BASE_URL}: {e}")
        ollama_available = False

    logger.info("RAG API startup complete")
    yield

    # Cleanup
    logger.info("Shutting down RAG API...")


# ---------------------------------------------------------------------------
# FastAPI app assembly
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
    description="FastAPI + Ollama + ChromaDB RAG pipeline",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _query_ollama(prompt: str, model: str = "llama3") -> str:
    """Query Ollama LLM via HTTP API and return the generated text."""
    import requests
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"Ollama HTTP error: {e}")
    return ""


def _retrieve_and_rerank(question: str, top_k: int = 5, rerank_top_k: int = 3) -> List[SourceDoc]:
    """Retrieve docs from ChromaDB and rerank using CrossEncoder."""
    if not embedding_model or not chroma_collection:
        return []

    # 1. Embed the question using the SentenceTransformer model
    query_embedding = embedding_model.encode(question, normalize_embeddings=True).tolist()

    # 2. Retrieve from ChromaDB
    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    docs: List[SourceDoc] = []
    if results["ids"] and results["ids"][0]:
        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{} for _ in ids]
        distances = results["distances"][0] if results.get("distances") else [1.0] * len(ids)

        for idx, doc_id in enumerate(ids):
            doc_content = documents[idx] if idx < len(documents) else ""
            meta = metadatas[idx] if idx < len(metadatas) else {}
            # Convert ChromaDB distance to similarity score
            similarity = 1 - distances[idx] if distances[idx] is not None else 1.0
            docs.append(
                SourceDoc(
                    id=doc_id,
                    content=doc_content,
                    metadata=meta,
                    similarity_score=similarity,
                )
            )

    if not docs:
        return []

    # 3. Rerank with CrossEncoder
    pairs = [(question, doc.content) for doc in docs]
    rerank_scores = reranker.predict(pairs)

    # Attach rerank scores and sort
    for doc, rer_score in zip(docs, rerank_scores):
        doc.similarity_score = float(rer_score)

    docs.sort(key=lambda d: d.similarity_score, reverse=True)
    return docs[:rerank_top_k]


def _build_prompt(question: str, sources: List[SourceDoc]) -> str:
    """Build the prompt for the LLM."""
    context_parts = []
    for i, src in enumerate(sources, 1):
        context_parts.append(
            f"[Source {i}] {src.content} (metadata: {src.metadata})"
        )
    context = "\n".join(context_parts) if context_parts else "No relevant sources found."

    prompt = f"""You are a helpful assistant. Answer the user's question using only the context provided below.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["rag"],
)
async def query_endpoint(request: QueryRequest):
    """POST /query - ask a question and get a RAG-generated answer with sources."""
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'question' field must be a non-empty string.",
        )

    logger.info(f"Received query: {request.question[:80]}...")

    try:
        # Retrieve & rerank relevant documents
        docs = _retrieve_and_rerank(request.question, top_k=5, rerank_top_k=3)

        if not docs:
            return QueryResponse(answer="No relevant documents found to answer the question.", sources=[])

        # Build prompt and query Ollama
        prompt = _build_prompt(request.question, docs)
        answer = _query_ollama(prompt, model="llama3")

        return QueryResponse(answer=answer, sources=docs)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error processing query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )