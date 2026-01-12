import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

@staticmethod
def load_rag_assets():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("students_faiss.index")

    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)

    return embedding_model, index, documents


def retrieve_context(query, embedding_model, index, documents, top_k=5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )
    _, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]
