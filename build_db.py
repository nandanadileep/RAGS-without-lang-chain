import requests
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

TEXT_FILE = "bible_texts.json"
VECTOR_FILE = "bible_vectors.npy"

def build():
    
    if os.path.exists(TEXT_FILE):
        print("   - Found existing text file. Skipping download.")
        with open(TEXT_FILE, 'r') as f:
            documents = json.load(f)
    else:
        print("   - Downloading Bible data...")
        url = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
        resp = requests.get(url)
        try:
            data = resp.json()
        except Exception:
            text = resp.content.decode('utf-8-sig')
            data = json.loads(text)
        
        documents = []
        for book in data:
            for c_num, chapter in enumerate(book['chapters'], 1):
                for v_num, text in enumerate(chapter, 1):
                    documents.append({
                        "ref": f"{book['name']} {c_num}:{v_num}",
                        "text": text
                    })
        
        with open(TEXT_FILE, "w", encoding="utf-8") as f:
            json.dump(documents, f, ensure_ascii=False)
        print(f"   - Saved {len(documents)} verses to {TEXT_FILE}")

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    texts = [doc['text'] for doc in documents]
    vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)
    
    np.save(VECTOR_FILE, vectors)
    print(f"   - Saved vectors to {VECTOR_FILE}")

if __name__ == "__main__":
    build()