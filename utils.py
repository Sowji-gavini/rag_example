import os
import uuid

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from groq import Groq

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL is missing in environment variables")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in environment variables")

COLLECTION_NAME = "rag_collection"

# ==========================================
# MODELS
# ==========================================

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_SIZE = 384

groq_client = Groq(api_key=GROQ_API_KEY)

# ✅ FIXED MODEL (no decommissioned model)
GROQ_MODEL = "llama-3.1-8b-instant"

# ==========================================
# QDRANT CLIENT
# ==========================================

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ==========================================
# PDF READER
# ==========================================

def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    ).strip()

# ==========================================
# CHUNKING
# ==========================================

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# ==========================================
# CREATE COLLECTION
# ==========================================

def create_collection():
    collections = qdrant.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME in names:
        print("Collection already exists")
        return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE
        )
    )

    print("Collection created")

# ==========================================
# STORE EMBEDDINGS
# ==========================================

def store_in_qdrant(chunks):
    points = []

    for chunk in chunks:
        vector = embedding_model.encode(chunk).tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk}
            )
        )

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print(f"Stored {len(points)} chunks")

# ==========================================
# SEARCH QDRANT (FIXED)
# ==========================================

def search_qdrant(query, limit=3):
    query_vector = embedding_model.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    ).points

    return [
        point.payload.get("text", "")
        for point in results
        if point.payload
    ]

# ==========================================
# ASK GROQ (RAG)
# ==========================================

def ask_groq(question, context_chunks):
    context = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict RAG QA assistant. "
                "Answer ONLY using provided context. "
                "If the answer is not in the context, say: "
                "'I could not find the answer in the document.'"
            )
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        }
    ]

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=512
    )

    return response.choices[0].message.content