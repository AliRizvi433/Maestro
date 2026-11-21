import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from paper_generator import _question_texts, exam_belongs_to_subject


JSON_DIR = Path(__file__).resolve().parent / "json_subs"


def get_top_repeated_questions(subject, top_n=5):
    occurrences = []
    file_path = JSON_DIR / f"{subject.upper()}.json"
    if not file_path.exists():
        return []

    with file_path.open(encoding="utf-8") as file:
        exams = json.load(file)

    for exam in exams:
        if not exam_belongs_to_subject(exam, subject.upper()):
            continue
        for text in _question_texts(exam):
            key = re.sub(r"\s+", " ", text).strip().lower()
            key = re.sub(r"[^a-z0-9\s]", "", key)
            occurrences.append(
                {
                    "text": text,
                    "key": key,
                    "year": str(exam.get("year", "")),
                }
            )

    if not occurrences:
        return []

    try:
        from vector import embeddings

        vectors = embeddings.embed_documents(
            [item["text"] for item in occurrences]
        )
    except Exception:
        vectors = None

    clusters = []
    for index, item in enumerate(occurrences):
        matching_cluster = None
        item_words = set(item["key"].split())

        for cluster in clusters:
            representative = cluster["representative"]
            representative_words = set(representative["key"].split())

            exact_or_char = SequenceMatcher(
                None,
                item["key"],
                representative["key"],
            ).ratio()
            shared_words = item_words & representative_words
            union_words = item_words | representative_words
            word_jaccard = (
                len(shared_words) / len(union_words)
                if union_words
                else 0
            )

            cosine = 0
            if vectors is not None:
                import math

                left = vectors[index]
                right = vectors[representative["index"]]
                dot = sum(a * b for a, b in zip(left, right))
                left_norm = math.sqrt(sum(a * a for a in left))
                right_norm = math.sqrt(sum(b * b for b in right))
                cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0

            if (
                exact_or_char >= 0.72
                or word_jaccard >= 0.58
                or cosine >= 0.78
            ):
                matching_cluster = cluster
                break

        if matching_cluster is None:
            matching_cluster = {
                "representative": {**item, "index": index},
                "items": [],
                "years": set(),
            }
            clusters.append(matching_cluster)

        matching_cluster["items"].append(item)
        if item["year"]:
            matching_cluster["years"].add(item["year"])

    repeated = [
        cluster
        for cluster in clusters
        if len(cluster["years"]) > 1
    ]
    repeated.sort(
        key=lambda cluster: (
            -len(cluster["years"]),
            -len(cluster["items"]),
            cluster["representative"]["text"].lower(),
        )
    )
    return [
        cluster["representative"]["text"]
        for cluster in repeated[:top_n]
    ]
