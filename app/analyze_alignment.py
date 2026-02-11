# app/analyze_alignment.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Base directory = parent of this file (..)
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_processed():
    with open(PROCESSED_DIR / "objectives.json", encoding="utf-8") as f:
        objectives = json.load(f)

    with open(PROCESSED_DIR / "actions.json", encoding="utf-8") as f:
        actions = json.load(f)

    return objectives, actions


def main():
    print("Loading processed data...")
    objectives, actions = load_processed()
    print(f"Loaded {len(objectives)} objectives and {len(actions)} actions.")

    # 1) Load embedding model
    print("Loading embedding model (this may download once)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good quality

    # 2) Prepare texts
    objective_texts = [o["full_text"] for o in objectives]
    action_texts = [a["full_text"] for a in actions]

    # Safety check
    if not objective_texts or not action_texts:
        print("No objectives or actions found. Exiting.")
        return

    # 3) Create embeddings
    print("Creating embeddings...")
    obj_emb = model.encode(objective_texts, normalize_embeddings=True)
    act_emb = model.encode(action_texts, normalize_embeddings=True)

    # 4) Similarity matrix: rows = actions, cols = objectives
    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity(act_emb, obj_emb)
    # shape: (num_actions, num_objectives)

    # 5) For each action, get most similar objective (Top-1)
    predicted_obj_indices = sim_matrix.argmax(axis=1)
    predicted_obj_ids = [objectives[i]["id"] for i in predicted_obj_indices]

    # 6) Evaluate vs declared objectives
    total = len(actions)
    correct_top1 = 0

    for action, pred_id in zip(actions, predicted_obj_ids):
        if pred_id in action.get("declared_objectives", []):
            correct_top1 += 1

    accuracy_top1 = correct_top1 / total if total else 0.0
    print(
        f"\nTop-1 alignment accuracy: {accuracy_top1:.2%} "
        f"({correct_top1}/{total} actions correctly matched)"
    )

    # Average similarity of predicted pairs
    pred_sims = sim_matrix[np.arange(len(actions)), predicted_obj_indices]
    print(f"Average similarity of predicted pairs: {pred_sims.mean():.3f}")

    # 7) Coverage: how many objectives have at least one strong action
    threshold = 0.60
    covered = 0

    for j in range(len(objectives)):
        sims_for_obj = sim_matrix[:, j]
        if (sims_for_obj >= threshold).any():
            covered += 1

    coverage_rate = covered / len(objectives) if objectives else 0.0
    print(
        f"Objective coverage (>= {threshold} similarity): "
        f"{coverage_rate:.2%} ({covered}/{len(objectives)})"
    )

    # 8) Save detailed similarity matrix to CSV
    rows = []
    for i, action in enumerate(actions):
        for j, obj in enumerate(objectives):
            rows.append({
                "action_id": action["id"],
                "objective_id": obj["id"],
                "similarity": float(sim_matrix[i, j])
            })

    df = pd.DataFrame(rows)
    csv_path = BASE_DIR / "alignment_matrix.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\nSaved detailed similarities to: {csv_path}")


if __name__ == "__main__":
    main()
