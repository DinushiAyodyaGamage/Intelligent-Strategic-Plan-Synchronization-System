# app/dashboard.py

from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- Base Paths ----------------

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ---------------- Ontology ----------------

CONCEPT_ONTOLOGY = {
    "employability": {
        "label": "Student careers & employability",
        "keywords": [
            "employability", "career", "careers", "placement", "internship",
            "industry", "employer", "job", "networking", "career fair",
            "work-ready", "work ready"
        ],
    },
    "student_success": {
        "label": "Student success, retention & support",
        "keywords": [
            "retention", "progression", "dropout", "drop-out", "support",
            "at-risk", "tutor", "mentoring", "transition", "induction"
        ],
    },
    "digital_learning": {
        "label": "Digital / blended learning",
        "keywords": [
            "digital", "online", "virtual learning", "blended", "vle",
            "e-assessment", "e assessment", "online labs", "technology-enhanced"
        ],
    },
    "research_innovation": {
        "label": "Research & innovation",
        "keywords": [
            "research", "publications", "publication", "citation",
            "grant", "funded project", "project", "centre", "center",
            "applied ai", "analytics"
        ],
    },
    "industry_partnerships": {
        "label": "Industry & community partnerships",
        "keywords": [
            "partnership", "partner", "community", "public sector",
            "collaboration", "collaborations", "employer", "alumni"
        ],
    },
    "data_infrastructure": {
        "label": "Data & analytics infrastructure",
        "keywords": [
            "data platform", "analytics", "dashboard", "reporting",
            "data infrastructure", "integrated data", "business intelligence"
        ],
    },
    "cybersecurity": {
        "label": "Cybersecurity & data protection",
        "keywords": [
            "cybersecurity", "security incident", "data protection",
            "gdpr", "breach", "firewall", "access control", "penetration testing"
        ],
    },
    "inclusion_wellbeing": {
        "label": "Inclusion, diversity & wellbeing",
        "keywords": [
            "inclusion", "inclusive", "diversity", "wellbeing", "well-being",
            "workload", "stress", "widening participation", "outreach",
            "under-represented"
        ],
    },
}


# ---------------- Concept Tagging ----------------

def tag_concepts(text: str):
    text_l = text.lower()
    tags = []
    for concept_key, spec in CONCEPT_ONTOLOGY.items():
        if any(kw in text_l for kw in spec["keywords"]):
            tags.append(concept_key)
    if not tags:
        tags.append("other")
    return tags


# ---------------- Concept-aware Suggestion ----------------

def concept_aware_suggestion(max_similarity: float, concepts: list):

    if max_similarity >= 0.75:
        alignment_strength = "STRONG"
        suggestion = "This objective is strongly supported by at least one action."
    elif max_similarity >= 0.60:
        alignment_strength = "MODERATE"
        suggestion = "This objective has moderate support, but alignment could be strengthened."
    else:
        alignment_strength = "WEAK"
        suggestion = "This objective appears weakly supported by current actions."

    concept_labels = [
        CONCEPT_ONTOLOGY[c]["label"] if c in CONCEPT_ONTOLOGY else c
        for c in concepts
    ]

    return (
        f"Alignment strength: {alignment_strength} "
        f"(max similarity = {max_similarity:.3f}).\n\n"
        f"Tagged strategic themes: {', '.join(concept_labels)}.\n\n"
        f"Recommendation: {suggestion}"
    )


# ---------------- Data Loading ----------------

@st.cache_data
def load_processed():
    with open(PROCESSED_DIR / "objectives.json", encoding="utf-8") as f:
        objectives = json.load(f)
    with open(PROCESSED_DIR / "actions.json", encoding="utf-8") as f:
        actions = json.load(f)
    return objectives, actions


@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------- Alignment Computation ----------------

@st.cache_data
def compute_alignment_and_tags():

    objectives, actions = load_processed()
    model = get_model()

    obj_texts = [o["full_text"] for o in objectives]
    act_texts = [a["full_text"] for a in actions]

    obj_emb = model.encode(obj_texts, normalize_embeddings=True)
    act_emb = model.encode(act_texts, normalize_embeddings=True)

    sim_matrix = cosine_similarity(act_emb, obj_emb)

    pred_idx = sim_matrix.argmax(axis=1)
    predicted_obj_ids = [objectives[i]["id"] for i in pred_idx]

    correct = sum(
        1 for action, pred_id in zip(actions, predicted_obj_ids)
        if pred_id in action["declared_objectives"]
    )
    top1_acc = correct / len(actions) if actions else 0.0

    pred_sims = sim_matrix[np.arange(len(actions)), pred_idx]
    avg_sim = float(pred_sims.mean()) if len(pred_sims) > 0 else 0.0

    threshold = 0.40
    covered = sum(
        1 for j in range(len(objectives))
        if (sim_matrix[:, j] >= threshold).any()
    )
    coverage = covered / len(objectives) if objectives else 0.0

    obj_concepts = {o["id"]: tag_concepts(o["full_text"]) for o in objectives}
    act_concepts = {a["id"]: tag_concepts(a["full_text"]) for a in actions}

    obj_summary = []
    for j, obj in enumerate(objectives):
        sims = sim_matrix[:, j]
        obj_summary.append({
            "objective_id": obj["id"],
            "title": obj["title"],
            "concepts": ", ".join(obj_concepts[obj["id"]]),
            "num_actions": len(actions),
            "max_similarity": float(sims.max()),
            "avg_similarity": float(sims.mean()),
        })

    return (
        sim_matrix,
        predicted_obj_ids,
        top1_acc,
        avg_sim,
        coverage,
        obj_summary,
        obj_concepts,
        act_concepts,
    )


# ---------------- FAISS Search ----------------

@st.cache_resource
def build_faiss_index():

    objectives, actions = load_processed()
    model = get_model()

    docs = []
    ids = []
    metadatas = []

    for obj in objectives:
        docs.append(obj["full_text"])
        ids.append(f"obj_{obj['id']}")
        metadatas.append({
            "type": "objective",
            "objective_id": obj["id"],
            "title": obj["title"],
        })

    for act in actions:
        docs.append(act["full_text"])
        ids.append(f"act_{act['id']}")
        metadatas.append({
            "type": "action",
            "action_id": act["id"],
            "declared_objectives": act["declared_objectives"],
        })

    embeddings = model.encode(docs, normalize_embeddings=True).astype("float32")
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, ids, metadatas, model


def semantic_search(query: str, top_k: int = 5):
    index, ids, metadatas, model = build_faiss_index()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(q_emb, top_k)

    results = []
    for score, i in zip(scores[0], idxs[0]):
        results.append({
            "id": ids[i],
            "score": float(score),
            "type": metadatas[i]["type"],
            "metadata": metadatas[i],
        })
    return results


# ---------------- Streamlit UI ----------------

def main():

    st.set_page_config(
        page_title="Intelligent Strategic Plan Synchronization System (ISPS)",
        layout="wide",
    )

    st.title("Intelligent Strategic Plan Synchronization System (ISPS)")
    st.caption("BrightPath University – Faculty of Computing & Data Sciences")

    objectives, actions = load_processed()
    (
        sim_matrix,
        predicted_obj_ids,
        top1_acc,
        avg_sim,
        coverage,
        obj_summary,
        obj_concepts,
        act_concepts,
    ) = compute_alignment_and_tags()

    view = st.sidebar.radio(
        "Select view",
        [
            "Overview",
            "Objective-wise alignment",
            "Action-wise alignment",
            "Ontology & concept view",
            "Semantic search",
        ],
    )

    # ---------------- Overview ----------------
    if view == "Overview":

        st.header("Overall Synchronization Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Top-1 alignment accuracy", f"{top1_acc:.1%}")
        col2.metric("Avg similarity (predicted pairs)", f"{avg_sim:.3f}")
        col3.metric("Objective coverage (≥ 0.60)", f"{coverage:.1%}")

        st.markdown("### Objective Support Summary")
        st.dataframe(pd.DataFrame(obj_summary), use_container_width=True)

    # ---------------- Objective-wise ----------------
    elif view == "Objective-wise alignment":

        st.header("Objective-wise Synchronization")

        obj_options = {f"{o['id']} – {o['title']}": idx for idx, o in enumerate(objectives)}
        selected_label = st.selectbox("Select a strategic objective", list(obj_options.keys()))
        j = obj_options[selected_label]
        selected_obj = objectives[j]

        st.subheader(f"{selected_obj['id']}: {selected_obj['title']}")

        concepts = obj_concepts[selected_obj["id"]]
        concept_labels = [
            CONCEPT_ONTOLOGY[c]["label"] if c in CONCEPT_ONTOLOGY else c
            for c in concepts
        ]
        st.markdown(f"*Concept tags:* {', '.join(concept_labels)}")

        sims_for_obj = sim_matrix[:, j]
        sorted_idx = np.argsort(-sims_for_obj)

        rows = []
        for rank, i in enumerate(sorted_idx, start=1):
            rows.append({
                "rank": rank,
                "action_id": actions[i]["id"],
                "declared_objectives": ", ".join(actions[i]["declared_objectives"]),
                "similarity": float(sims_for_obj[i]),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        max_sim = float(sims_for_obj.max())
        st.markdown("#### System suggestion")
        st.info(concept_aware_suggestion(max_sim, concepts))

    # ---------------- Semantic Search ----------------
    elif view == "Semantic search":

        st.header("Semantic Search")

        query = st.text_input("Enter search query")
        top_k = st.slider("Number of results", 3, 10, 5)

        if st.button("Search"):
            results = semantic_search(query, top_k)
            for res in results:
                st.write(f"{res['id']} (score: {res['score']:.3f})")


if __name__ == "__main__":
    main()
