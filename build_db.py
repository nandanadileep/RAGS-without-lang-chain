import requests
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

TEXT_FILE = "bible_texts.json"
VECTOR_FILE = "bible_vectors.npy"
ASSETS_DIR = "assets"

# High-Quality Art URLs
IMAGES = {
    "creation.jpg": "https://upload.wikimedia.org/wikipedia/commons/5/5b/Michelangelo_-_Creation_of_Adam_%28cropped%29.jpg",
    "supper.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/48/The_Last_Supper_-_Leonardo_Da_Vinci_-_High_Resolution_32x16.jpg",
    "prodigal.jpg": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Rembrandt_Harmensz_van_Rijn_-_Return_of_the_Prodigal_Son_-_Google_Art_Project.jpg",
    "annunciation.jpg": "https://upload.wikimedia.org/wikipedia/commons/9/93/Fra_Angelico_-_The_Annunciation_-_WGA00452.jpg",
    "crucifixion.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/31/El_Greco_-_The_Crucifixion_-_WGA10474.jpg"
}

def download_assets():
    """Downloads images to a local folder so they load instantly."""
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
        print(f"📁 Created '{ASSETS_DIR}' folder.")

    print("🖼️  Downloading Art Assets...")
    for filename, url in IMAGES.items():
        filepath = os.path.join(ASSETS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"   - Downloading {filename}...")
            try:
                content = requests.get(url).content
                with open(filepath, "wb") as f:
                    f.write(content)
            except Exception as e:
                print(f"   ❌ Failed to download {filename}: {e}")
        else:
            print(f"   - {filename} already exists.")

def build():
    # 1. Download Images FIRST
    download_assets()

    # 2. Download Text
    if os.path.exists(TEXT_FILE):
        print("✅ Found Bible text file.")
        with open(TEXT_FILE, 'r') as f:
            documents = json.load(f)
    else:
        print("📥 Downloading Bible data...")
        url = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"
        resp = requests.get(url)
        # Handle potential encoding BOM issues
        try:
            data = resp.json()
        except:
            data = json.loads(resp.content.decode('utf-8-sig'))
        
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
        print(f"✅ Saved {len(documents)} verses.")

    # 3. Create Vectors
    if not os.path.exists(VECTOR_FILE):
        print("🧠 Creating Vectors (This takes time)...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [doc['text'] for doc in documents]
        vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)
        np.save(VECTOR_FILE, vectors)
        print("✅ Index saved.")
    else:
        print("✅ Vectors already exist.")

if __name__ == "__main__":
    build()