# app/ingest.py

import json
import re
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


# -----------------------------
# Utilities
# -----------------------------
def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


# -----------------------------
# Strategic Objectives (S1–S5)
# -----------------------------
def extract_objectives(strategic_text: str):
    """
    Extract Strategic Objectives like:
    Strategic Objective S1: Enhance Student Learning Experience
    """
    pattern = re.compile(
        r"Strategic Objective\s+(S\d)\s*:\s*(.+)",
        re.IGNORECASE
    )

    objectives = []

    for line in strategic_text.splitlines():
        match = pattern.search(line)
        if match:
            obj_id = match.group(1).strip()      # S1
            title = match.group(2).strip()

            objectives.append({
                "id": obj_id,
                "title": title,
                "full_text": title
            })

    return objectives


# -----------------------------
# Actions (A1–A50)
# -----------------------------
def extract_actions(action_text: str):
    """
    Extract actions written in table-like format:
    A1
    S1
    Description...
    """
    action_id_pattern = re.compile(r"^A\d+$")
    objective_pattern = re.compile(r"^S\d+$")

    actions = []
    lines = [l.strip() for l in action_text.splitlines() if l.strip()]

    i = 0
    while i < len(lines):
        # Detect Action ID (A1, A2, ..., A50)
        if action_id_pattern.match(lines[i]):
            action_id = lines[i]

            # Next line should be Strategic Objective (S1–S5)
            if i + 1 < len(lines) and objective_pattern.match(lines[i + 1]):
                related_obj = lines[i + 1]
                description = lines[i + 2] if i + 2 < len(lines) else ""

                actions.append({
                    "id": action_id,
                    "title": description[:120],
                    "full_text": description,
                    "declared_objectives": [related_obj]
                })

                i += 3
                continue

        i += 1

    return actions


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    strategic_path = DATA_DIR / "strategic_plan" / "strategic_plan.txt"
    action_path = DATA_DIR / "action_plan" / "action_plan.txt"

    print("Loading files...")
    strategic_text = load_text(strategic_path)
    action_text = load_text(action_path)

    print("Extracting objectives...")
    objectives = extract_objectives(strategic_text)

    print("Extracting actions...")
    actions = extract_actions(action_text)

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(exist_ok=True)

    with open(processed_dir / "objectives.json", "w", encoding="utf-8") as f:
        json.dump(objectives, f, indent=2, ensure_ascii=False)

    with open(processed_dir / "actions.json", "w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extracted {len(objectives)} objectives and {len(actions)} actions.")


if __name__ == "__main__":
    main()
