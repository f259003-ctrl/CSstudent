import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_data(path="cs_students.csv"):
    df = pd.read_csv(path)
    documents = []

    for _, row in df.iterrows():
        doc = f"""
        GPA: {row['GPA']}
        Major: {row['Major']}
        Interested Domain: {row['Interested Domain']}
        Projects: {row['Projects']}
        Skills:
          Python: {row['Python']}
          SQL: {row['SQL']}
          Java: {row['Java']}
        Future Career: {row['Future Career']}
        """
        documents.append(doc)

    return df, documents


def build_faiss(documents):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def retrieve_context(query, model, index, documents, k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    _, idx = index.search(q_emb, k)
    return [documents[i] for i in idx[0]]


def rag_predict_label(i, model, index, documents, df, k=5):
    q = documents[i]
    q_emb = model.encode([q], convert_to_numpy=True)
    _, idx = index.search(q_emb, k)
    careers = [df.iloc[j]["Future Career"] for j in idx[0]]
    return max(set(careers), key=careers.count)
