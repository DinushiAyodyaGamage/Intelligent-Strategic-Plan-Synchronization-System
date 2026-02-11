# app/faiss_search_demo.py

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# -----------------------------
# Load processed data
# -----------------------------
def load_data():
    with open(PROCESSED_DIR / "objectives.json", encoding="utf-8") as f:
        objectives = json.load(f)
    with open(PROCESSED_DIR / "actions.json", encoding="utf-8") as f:
        actions = json.load(f)
    return objectives, actions


# -----------------------------
# Build FAISS index
# -----------------------------
def build_faiss_index():
    objectives, actions = load_data()

    docs = []
    metadatas = []
    ids = []

    # Add objectives
    for obj in objectives:
        docs.append(obj["full_text"])
        metadatas.append({
            "type": "objective",
            "objective_id": obj["id"],
            "title": obj["title"],
        })
        ids.append(f"obj_{obj['id']}")

    # Add actions
    for act in actions:
        docs.append(act["full_text"])
        metadatas.append({
            "type": "action",
            "action_id": act["id"],
            "declared_objectives": act["declared_objectives"],
        })
        ids.append(f"act_{act['id']}")

    # Load same embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings for FAISS index...")
    embeddings = model.encode(
        docs,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized vectors)
    index.add(embeddings)

    print(f"✅ FAISS index built with {index.ntotal} vectors.")
    return model, index, ids, metadatas


# -----------------------------
# Demo query
# -----------------------------
def demo_query():
    model, index, ids, metadatas = build_faiss_index()

    query = "actions that improve graduate employability and student careers"
    print(f"\n🔍 Query: {query}\n")

    q_emb = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    k = 5
    scores, idxs = index.search(q_emb, k)

    for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), start=1):
        meta = metadatas[i]
        doc_id = ids[i]

        print(f"Rank #{rank}")
        print(f"ID: {doc_id}")
        print(f"Type: {meta['type']}")
        print(f"Score (cosine similarity): {score:.3f}")
        print(f"Metadata: {meta}")
        print("-" * 60)


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    demo_query()
