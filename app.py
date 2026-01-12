import streamlit as st
from transformers import pipeline
from sklearn.metrics import accuracy_score
from rag_utils import (
    load_data,
    build_faiss,
    retrieve_context,
    rag_predict_label
)

st.set_page_config(page_title="Student Career RAG", layout="wide")
st.title("🎓 Student Career Prediction using RAG (FLAN-T5)")

@st.cache_resource
def load_models():
    df, documents = load_data()
    embed_model, index = build_faiss(documents)
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return df, documents, embed_model, index, llm

df, documents, embed_model, index, llm = load_models()

query = st.text_input("Ask a future career prediction question:")

if st.button("Predict"):
    context = retrieve_context(query, embed_model, index, documents)

    prompt = f"""
You are a career prediction assistant.

Use the following student profiles to reason and predict future outcomes.

Student Data:
{context}

Question:
{query}

Answer with reasoning.
"""

    result = llm(prompt)[0]["generated_text"]
    st.success(result)

st.divider()
st.subheader("📊 RAG Retrieval Accuracy")

if st.button("Calculate Accuracy"):
    predictions = [
        rag_predict_label(i, embed_model, index, documents, df)
        for i in range(len(df))
    ]
    accuracy = accuracy_score(df["Future Career"], predictions)
    st.info(f"RAG Accuracy: {accuracy:.2%}")
