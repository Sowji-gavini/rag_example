# app.py

import streamlit as st

from utils import (
    read_pdf,
    chunk_text,
    create_collection,
    store_in_qdrant,
    search_qdrant,
    ask_groq
)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Minimal RAG Chatbot",
    page_icon="🤖"
)

st.title("📄 Minimal RAG Chatbot")
st.write("Upload a PDF and ask questions about it.")

# ==========================================
# INIT COLLECTION (run once)
# ==========================================

create_collection()

# ==========================================
# SESSION STATE INIT
# ==========================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ==========================================
# PROCESS PDF ONLY ONCE
# ==========================================

if uploaded_file is not None and not st.session_state.processed:

    with st.spinner("Processing PDF..."):

        text = read_pdf(uploaded_file)
        chunks = chunk_text(text)

        store_in_qdrant(chunks)

        st.session_state.processed = True
        st.session_state.chunks_count = len(chunks)

    st.success("PDF processed successfully!")

st.write(f"Chunks in memory: {st.session_state.chunks_count}")

# ==========================================
# QUESTION INPUT
# ==========================================

question = st.text_input("Ask a question")

# ==========================================
# ANSWER GENERATION
# ==========================================

if question:

    with st.spinner("Generating answer..."):

        relevant_chunks = search_qdrant(question)
        answer = ask_groq(question, relevant_chunks)

    st.subheader("Answer")
    st.write(answer)

    # DEBUG VIEW
    with st.expander("Retrieved Chunks"):
        for i, chunk in enumerate(relevant_chunks):
            st.write(f"Chunk {i+1}")
            st.write(chunk)
            st.divider()

# ==========================================
# RESET BUTTON (IMPORTANT ADDITION)
# ==========================================

if st.button("Reset / Upload New PDF"):
    st.session_state.processed = False
    st.session_state.chunks_count = 0
    st.rerun()