import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

TEXT_FILE = "bible_texts.json"
VECTOR_FILE = "bible_vectors.npy"

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

def load_data():
    if not os.path.exists(TEXT_FILE) or not os.path.exists(VECTOR_FILE):
        raise FileNotFoundError(" Database missing! Run 'python build_db.py' first.")
    
    print("⚡ Loading Database from disk...")
    
    with open(TEXT_FILE, "r") as f:
        docs = json.load(f)
        
    vectors = np.load(VECTOR_FILE)
    
    return docs, vectors

def find_verses(query, documents, vectors, top_k=5):
    embedder = get_embedder()
    query_vec = embedder.encode([query])[0]
    
    scores = np.dot(vectors, query_vec) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)
    )
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        doc = documents[idx].copy()
        doc['score'] = float(scores[idx])
        results.append(doc)
        
    return results