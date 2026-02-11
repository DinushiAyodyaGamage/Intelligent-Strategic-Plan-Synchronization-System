# app/evaluate_metrics.py

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_data():
    with open(PROCESSED_DIR / "objectives.json", encoding="utf-8") as f:
        objectives = json.load(f)

    with open(PROCESSED_DIR / "actions.json", encoding="utf-8") as f:
        actions = json.load(f)

    return objectives, actions


def main():
    objectives, actions = load_data()

    if not objectives or not actions:
        print("No objectives or actions found. Please run ingest.py first.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding objectives...")
    obj_texts = [o["full_text"] for o in objectives]
    obj_emb = model.encode(obj_texts, normalize_embeddings=True)

    print("Encoding actions...")
    act_texts = [a["full_text"] for a in actions]
    act_emb = model.encode(act_texts, normalize_embeddings=True)

    print("Computing similarity matrix...")
    sim_matrix = cosine_similarity(act_emb, obj_emb)

    # -----------------------------
    # Top-1 and Top-3 Accuracy
    # -----------------------------
    num_actions = len(actions)
    top1_correct = 0
    top3_correct = 0

    for i, action in enumerate(actions):
        sims = sim_matrix[i, :]
        top_indices = np.argsort(-sims)

        top1_idx = top_indices[0]
        top3_idx = top_indices[:3]

        top1_id = objectives[top1_idx]["id"]
        top3_ids = [objectives[j]["id"] for j in top3_idx]

        if top1_id in action["declared_objectives"]:
            top1_correct += 1

        if any(obj_id in action["declared_objectives"] for obj_id in top3_ids):
            top3_correct += 1

    top1_acc = top1_correct / num_actions
    top3_acc = top3_correct / num_actions

    print("\nRetrieval Accuracy:")
    print(f"Top-1 accuracy: {top1_acc:.2%} ({top1_correct}/{num_actions})")
    print(f"Top-3 accuracy: {top3_acc:.2%} ({top3_correct}/{num_actions})")

    # -----------------------------
    # Precision / Recall per Objective
    # -----------------------------
    threshold = 0.35
    print(f"\nPer-objective precision/recall (threshold = {threshold}):")

    precisions = []
    recalls = []

    for j, obj in enumerate(objectives):
        obj_id = obj["id"]
        sims_for_obj = sim_matrix[:, j]

        pred_pos_indices = np.where(sims_for_obj >= threshold)[0]
        true_pos_indices = [
            i for i, a in enumerate(actions)
            if obj_id in a["declared_objectives"]
        ]

        pred_pos = set(pred_pos_indices)
        true_pos = set(true_pos_indices)

        tp = len(pred_pos & true_pos)
        fp = len(pred_pos - true_pos)
        fn = len(true_pos - pred_pos)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)

        print(
            f"{obj_id}: precision={precision:.2f}, "
            f"recall={recall:.2f}, TP={tp}, FP={fp}, FN={fn}"
        )

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)

    macro_f1 = (
        2 * macro_precision * macro_recall /
        (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    print("\nMacro-averaged metrics:")
    print(f"Precision: {macro_precision:.2f}")
    print(f"Recall:    {macro_recall:.2f}")
    print(f"F1-score:  {macro_f1:.2f}")


if __name__ == "__main__":
    main()
