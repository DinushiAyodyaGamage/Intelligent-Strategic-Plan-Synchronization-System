# app/dashboard.py

from pathlib import Path
import json

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# --------- Simple Ontology / Concept Definitions ---------

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


def tag_concepts(text: str):
    text_l = text.lower()
    tags = []
    for concept_key, spec in CONCEPT_ONTOLOGY.items():
        if any(kw in text_l for kw in spec["keywords"]):
            tags.append(concept_key)
    if not tags:
        tags.append("other")
    return tags


# --------- Data & Model Loading ---------

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

    correct = 0
    for action, pred_id in zip(actions, predicted_obj_ids):
        if pred_id in action["declared_objectives"]:
            correct += 1
    top1_acc = correct / len(actions) if actions else 0.0

    pred_sims = sim_matrix[np.arange(len(actions)), pred_idx]
    avg_sim = float(pred_sims.mean()) if len(pred_sims) > 0 else 0.0

    threshold = 0.50
    covered = 0
    for j in range(len(objectives)):
        if (sim_matrix[:, j] >= threshold).any():
            covered += 1
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
        meta = metadatas[i]
        results.append({
            "id": ids[i],
            "score": float(score),
            "type": meta["type"],
            "metadata": meta,
        })
    return results

def concept_aware_suggestion(max_similarity: float, concepts: list):
    """
    Provide interpretation and suggestion based on similarity score
    and ontology concept tags.
    """

    # Interpret similarity strength
    if max_similarity >= 0.75:
        alignment_strength = "strong"
        suggestion = "This objective is strongly supported by at least one action."
    elif max_similarity >= 0.60:
        alignment_strength = "moderate"
        suggestion = "This objective has moderate support, but alignment could be strengthened."
    else:
        alignment_strength = "weak"
        suggestion = "This objective appears weakly supported by current actions."

    # Add concept-based explanation
    concept_labels = [
        CONCEPT_ONTOLOGY[c]["label"] if c in CONCEPT_ONTOLOGY else c
        for c in concepts
    ]

    concept_text = ", ".join(concept_labels)

    return (
        f"Alignment strength: **{alignment_strength.upper()}** "
        f"(max similarity = {max_similarity:.3f}).\n\n"
        f"Tagged strategic themes: {concept_text}.\n\n"
        f"Recommendation: {suggestion}"
    )

# --------- Streamlit App UI ---------

def main():

    st.set_page_config(
        page_title="Intelligent Strategic Plan Synchronization System (ISPS)",
        layout="wide",
    )

    st.title("Intelligent Strategic Plan Synchronization System (ISPS)")
    st.caption("BrightPath University University – Faculty of Computing & Data Sciences")

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


# --------- Streamlit App UI ---------

def main():
    st.set_page_config(
        page_title="Intelligent Strategic Plan Synchronization System (ISPS)",
        layout="wide",
    )

    st.title("Intelligent Strategic Plan Synchronization System (ISPS)")
    st.caption("BrightPath University")

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

    if view == "Overview":
        st.header("Overall Synchronization Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Top-1 alignment accuracy", f"{top1_acc:.1%}")
        col2.metric("Avg similarity (predicted pairs)", f"{avg_sim:.3f}")
        col3.metric("Objective coverage (≥ 0.50)", f"{coverage:.1%}")

        st.markdown("### Objective Support Summary")
        df_obj = pd.DataFrame(obj_summary)
        st.dataframe(df_obj, use_container_width=True)

        st.markdown(
            "Interpretation: max_similarity indicates how strongly at least one action "
            "aligns with each objective; avg_similarity shows overall support. "
            "concepts show how objectives map onto key strategic themes in the ontology."
        )

    elif view == "Objective-wise alignment":
        st.header("Objective-wise Synchronization")

        obj_options = {f"{o['id']} – {o['title']}": idx for idx, o in enumerate(objectives)}
        selected_label = st.selectbox("Select a strategic objective", list(obj_options.keys()))
        j = obj_options[selected_label]
        selected_obj = objectives[j]

        st.subheader(f"{selected_obj['id']}: {selected_obj['title']}")
        concepts = obj_concepts[selected_obj["id"]]
        concept_labels = [
            CONCEPT_ONTOLOGY[c]["label"] if c in CONCEPT_ONTOLOGY else c for c in concepts
        ]
        st.markdown(f"*Concept tags:* {', '.join(concept_labels)}")

        sims_for_obj = sim_matrix[:, j]
        # Sort actions by similarity descending
        sorted_idx = np.argsort(-sims_for_obj)

        rows = []
        for rank, i in enumerate(sorted_idx, start=1):
            action = actions[i]
            rows.append({
                "rank": rank,
                "action_id": action["id"],
                "declared_objectives": ", ".join(action["declared_objectives"]),
                "similarity": float(sims_for_obj[i]),
            })
        df_actions = pd.DataFrame(rows)

        st.markdown("#### Actions aligned with this objective (sorted by similarity)")
        st.dataframe(df_actions, use_container_width=True)

        max_sim = float(sims_for_obj.max())
        st.markdown("#### System suggestion")
        st.info(concept_aware_suggestion(max_sim, concepts))

    elif view == "Action-wise alignment":
        st.header("Action-wise Synchronization")

        act_options = {f"{a['id']}": idx for idx, a in enumerate(actions)}
        selected_act_id = st.selectbox("Select an action (by ID)", list(act_options.keys()))
        i = act_options[selected_act_id]
        selected_action = actions[i]

        st.subheader(f"Action {selected_action['id']}")
        st.markdown(
            f"*Declared objectives:* "
            f"{', '.join(selected_action['declared_objectives']) or 'None'}"
        )

        # Concepts for this action
        a_concepts = act_concepts[selected_action["id"]]
        concept_labels = [
            CONCEPT_ONTOLOGY[c]["label"] if c in CONCEPT_ONTOLOGY else c for c in a_concepts
        ]
        st.markdown(f"*Concept tags:* {', '.join(concept_labels)}")

        # Similarities to all objectives for this action
        sims_for_action = sim_matrix[i, :]
        rows = []
        for j, obj in enumerate(objectives):
            rows.append({
                "objective_id": obj["id"],
                "title": obj["title"],
                "similarity": float(sims_for_action[j]),
                "is_predicted": (obj["id"] == predicted_obj_ids[i]),
                "is_declared": (obj["id"] in selected_action["declared_objectives"]),
            })
        df = pd.DataFrame(rows)
        df_sorted = df.sort_values(by="similarity", ascending=False)

        st.markdown("#### Alignment with strategic objectives")
        st.dataframe(df_sorted, use_container_width=True)

        st.markdown("*Notes:*")
        st.markdown(
            "- is_predicted = True indicates the objective with highest similarity.\n"
            "- Compare this with is_declared to see if the action is mapped correctly.\n"
            "- Concept tags help explain why an action is aligned with particular objectives."
        )

    elif view == "Ontology & concept view":
        st.header("Ontology & Concept-Based Synchronization")

        concept_keys = list(CONCEPT_ONTOLOGY.keys()) + ["other"]
        concept_labels = {
            k: (CONCEPT_ONTOLOGY[k]["label"] if k in CONCEPT_ONTOLOGY else "Other / uncategorised")
            for k in concept_keys
        }

        selected_concept_key = st.selectbox(
            "Select a concept",
            options=concept_keys,
            format_func=lambda k: concept_labels[k],
        )

        st.markdown(f"### Concept: {concept_labels[selected_concept_key]}")

        # Objectives with this concept
        obj_rows = []
        for summary in obj_summary:
            obj_id = summary["objective_id"]
            if selected_concept_key in obj_concepts[obj_id]:
                obj_rows.append(summary)

        # Actions with this concept
        act_rows = []
        for a in actions:
            if selected_concept_key in act_concepts[a["id"]]:
                act_rows.append({
                    "action_id": a["id"],
                    "declared_objectives": ", ".join(a["declared_objectives"]),
                    "concepts": ", ".join(act_concepts[a["id"]]),
                })

        col1, col2 = st.columns(2)
        col1.metric("Objectives with this concept", len(obj_rows))
        col2.metric("Actions with this concept", len(act_rows))

        st.markdown("#### Objectives tagged with this concept")
        if obj_rows:
            st.dataframe(pd.DataFrame(obj_rows), use_container_width=True)
        else:
            st.write("No strategic objectives were tagged with this concept.")

        st.markdown("#### Actions tagged with this concept")
        if act_rows:
            st.dataframe(pd.DataFrame(act_rows), use_container_width=True)
        else:
            st.write("No actions in the current plan were tagged with this concept.")

        st.markdown(
            "Use this view to see whether important strategic concepts are adequately reflected "
            "in the action plan. For example, a concept that appears in several objectives but "
            "very few actions may indicate an implementation gap."
        )

    elif view == "Semantic search":
        st.header("Semantic Search over Strategic & Action Plan")

        query = st.text_input(
            "Enter a search query (e.g., 'improve digital learning' or 'support staff wellbeing')",
            value="improve graduate employability and student careers",
        )
        top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)

        if st.button("Search"):
            with st.spinner("Searching using FAISS vector index..."):
                results = semantic_search(query, top_k=top_k)

            st.markdown("#### Top results")
            for res in results:
                meta = res["metadata"]
                score = res["score"]

                if res["type"] == "objective":
                    st.write(
                        f"*Objective {meta['objective_id']}* – {meta['title']}  "
                        f"(similarity: {score:.3f})"
                    )
                else:
                    st.write(
                        f"*Action {meta['action_id']}*  "
                        f"(similarity: {score:.3f}, "
                        f"declared: {', '.join(meta['declared_objectives'])})"
                    )
            st.markdown("---")
            st.caption("Powered by sentence-transformer embeddings + FAISS index.")


if __name__ == "__main__":
    main()
