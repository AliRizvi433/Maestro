import json
import random
from pathlib import Path


JSON_DIR = Path(__file__).resolve().parent / "json_subs"


def _question_texts(exam):
    texts = []
    for block in exam.get("questions", []):
        texts.extend(
            part.get("text", "").strip()
            for part in block.get("question_parts", [])
            if part.get("text", "").strip()
        )
    for section in exam.get("sections", []):
        for block in section.get("questions", []):
            texts.extend(
                part.get("text", "").strip()
                for part in block.get("sub_questions", [])
                if part.get("text", "").strip()
            )
    return texts


def generate_sample_paper(subject, year_filter=None, exam_type_filter=None):
    file_path = JSON_DIR / f"{subject.upper()}.json"
    if not file_path.exists():
        return f"No exam data found for subject: {subject}"

    with file_path.open(encoding="utf-8") as file:
        exams = json.load(file)

    question_pool = []
    for exam in exams:
        if year_filter and str(exam.get("year")) != str(year_filter):
            continue
        if exam_type_filter and exam.get("exam_type", "").lower() != exam_type_filter.lower():
            continue
        question_pool.extend(_question_texts(exam))

    if not question_pool:
        return "No actual questions available for the selected filter."

    sample_questions = random.sample(question_pool, min(8, len(question_pool)))
    paper = [f"Sample Paper for {subject.upper()}", "", "Section A (Compulsory)"]
    paper.extend(f"{index}. {question}" for index, question in enumerate(sample_questions[:4], 1))
    paper.extend(["", "Section B (Answer any 3)"])
    paper.extend(
        f"{index}. {question}"
        for index, question in enumerate(sample_questions[4:], 5)
    )
    return "\n".join(paper)
