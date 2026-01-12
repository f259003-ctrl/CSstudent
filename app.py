import streamlit as st
from transformers import pipeline
from rag_utils import load_rag_assets, retrieve_context

st.set_page_config(page_title="Student Career RAG", layout="wide")
st.title("🎓 Student Career Prediction (RAG + FLAN-T5)")

@st.cache_resource
def load_models():
    embedding_model, index, documents = load_rag_assets()

    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    return embedding_model, index, documents, llm


embedding_model, index, documents, llm = load_models()

query = st.text_input(
    "Ask a future career prediction question:",
    placeholder="e.g., Predict career for students strong in Python & AI"
)

if st.button("Predict"):
    context = retrieve_context(
        query, embedding_model, index, documents
    )

    prompt = f"""
You are a career prediction assistant.

Use the student profiles below to reason and predict future outcomes.

Student Data:
{context}

Question:
{query}

Provide a clear, future-oriented prediction.
"""

    result = llm(prompt)[0]["generated_text"]
    st.success(result)
