import json
import random
from pathlib import Path


JSON_DIR = Path(__file__).resolve().parent / "json_subs"


def exam_belongs_to_subject(exam, subject):
    subject_name = str(exam.get("subject_name", "")).lower()
    subject_code = str(exam.get("subject_code", "")).upper()

    if subject == "AI":
        return "artificial intelligence" in subject_name
    if subject == "ML":
        return "machine learning" in subject_name or subject_name in {"ml", "machine-learning"}
    if subject == "CC":
        return "cloud" in subject_name or "cloud computing" in subject_name
    if subject == "UHV":
        return "human values" in subject_name or "universal human" in subject_name
    return False


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
        if not exam_belongs_to_subject(exam, subject.upper()):
            continue
        if year_filter and str(exam.get("year")) != str(year_filter):
            continue
        if exam_type_filter and exam.get("exam_type", "").lower() != exam_type_filter.lower():
            continue
        question_pool.extend(_question_texts(exam))

    if not question_pool:
        return "No actual questions available for the selected filter."

    target_count = min(24, len(question_pool))
    sample_questions = random.sample(question_pool, target_count)

    section_a_count = min(8, target_count)
    section_b_count = min(8, max(0, target_count - section_a_count))
    section_c_count = max(0, target_count - section_a_count - section_b_count)

    paper = [
        f"Sample Paper for {subject.upper()}",
        "",
        "Section A (Compulsory)",
        "Answer all questions.",
    ]
    paper.extend(
        f"{index}. {question}"
        for index, question in enumerate(sample_questions[:section_a_count], 1)
    )

    paper.extend(["", "Section B (Answer any 5)", ""])
    section_b_start = section_a_count
    section_b_end = section_a_count + section_b_count
    paper.extend(
        f"{index}. {question}"
        for index, question in enumerate(
            sample_questions[section_b_start:section_b_end],
            section_a_count + 1,
        )
    )

    if section_c_count:
        paper.extend(["", "Section C (Answer any 3)", ""])
        section_c_start = section_b_end
        paper.extend(
            f"{index}. {question}"
            for index, question in enumerate(
                sample_questions[section_c_start:],
                section_a_count + section_b_count + 1,
            )
        )
    return "\n".join(paper)
