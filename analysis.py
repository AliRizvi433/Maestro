import json
import re
from collections import defaultdict
from pathlib import Path

from paper_generator import _question_texts


JSON_DIR = Path(__file__).resolve().parent / "json_subs"


def get_top_repeated_questions(subject, top_n=5):
    occurrences = defaultdict(set)
    display_text = {}
    file_path = JSON_DIR / f"{subject.upper()}.json"
    if not file_path.exists():
        return []

    with file_path.open(encoding="utf-8") as file:
        exams = json.load(file)

    for exam in exams:
        for text in _question_texts(exam):
            key = re.sub(r"\s+", " ", text).strip().lower()
            occurrences[key].add(str(exam.get("year", "")))
            display_text.setdefault(key, text)

    repeated = sorted(
        ((key, years) for key, years in occurrences.items() if len(years) > 1),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return [display_text[key] for key, _ in repeated[:top_n]]
