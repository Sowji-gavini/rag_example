import streamlit as st

st.title("Debug Check")
st.write("QDRANT_URL:", st.secrets.get("QDRANT_URL", "NOT FOUND"))
st.write("QDRANT_API_KEY length:", len(st.secrets.get("QDRANT_API_KEY", "")))
st.write("GROQ_API_KEY length:", len(st.secrets.get("GROQ_API_KEY", "")))
