import os
import uuid

import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from groq import Groq

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

COLLECTION_NAME = "rag_collection"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_SIZE = 384

groq_client = Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
except Exception as e:
    st.error(f"Qdrant connection failed: {e}")
    raise

def read_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def create_collection():
    try:
        collections = qdrant.get_collections().collections
        names = [c.name for c in collections]
        if COLLECTION_NAME not in names:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)
            )
    except Exception as e:
        st.error(f"Qdrant error: {e}")
        raise

def store_in_qdrant(chunks):
    points = []
    for chunk in chunks:
        vector = embedding_model.encode(chunk).tolist()
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk}))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

def search_qdrant(query, limit=3):
    query_vector = embedding_model.encode(query).tolist()
    results = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=limit, with_payload=True).points
    return [point.payload.get("text", "") for point in results if point.payload]

def ask_groq(question, context_chunks):
    context = "\n\n".join(context_chunks)
    messages = [
        {"role": "system", "content": "You are a strict RAG QA assistant. Answer ONLY using provided context. If the answer is not in the context, say: 'I could not find the answer in the document.'"},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"}
    ]
    response = groq_client.chat.completions.create(model=GROQ_MODEL, messages=messages, temperature=0.2, max_tokens=512)
    return response.choices[0].message.content
