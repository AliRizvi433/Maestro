import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


ROOT_DIR = Path(__file__).resolve().parent
JSON_DIR = ROOT_DIR / "json_subs"
DB_DIR = ROOT_DIR / "chroma_question_db"
COLLECTION_NAME = "question_parts"

load_dotenv(ROOT_DIR / ".env")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Set GEMINI_API_KEY in the environment or in a local .env file before starting Maestro."
    )


def _metadata_without_none(values):
    return {key: value for key, value in values.items() if value is not None}


def _question_documents(exam, subject):
    common_metadata = _metadata_without_none(
        {
            "year": str(exam.get("year", "")),
            "exam_type": str(exam.get("exam_type", "")),
            "semester": exam.get("semester"),
            "programme": exam.get("programme"),
            "branch": exam.get("branch"),
            "subject": subject,
            "subject_name": exam.get("subject_name"),
            "subject_code": exam.get("subject_code"),
        }
    )

    documents = []
    for block in exam.get("questions", []):
        for part in block.get("question_parts", []):
            text = str(part.get("text", "")).strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            **common_metadata,
                            "question_number": str(block.get("question_number", "")),
                            "question_section": block.get("question_section", ""),
                            "part_id": str(part.get("part_id", "")),
                            "marks": part.get("marks", 0) or 0,
                        },
                    )
                )

    for section in exam.get("sections", []):
        for block in section.get("questions", []):
            for part in block.get("sub_questions", []):
                text = str(part.get("text", "")).strip()
                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                **common_metadata,
                                "question_number": str(block.get("question_number", "")),
                                "question_section": section.get("section_name", ""),
                                "part_id": "",
                                "marks": block.get("marks_per_sub_question", 0) or 0,
                            },
                        )
                    )
    return documents


def load_question_documents():
    documents = []
    for file_path in sorted(JSON_DIR.glob("*.json")):
        with file_path.open(encoding="utf-8") as file:
            exams = json.load(file)
        subject = file_path.stem.upper()
        for exam in exams:
            documents.extend(_question_documents(exam, subject))
    return documents


embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
    api_key=api_key,
)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)

def build_vector_index():
    documents = load_question_documents()
    if not documents:
        raise RuntimeError(f"No question data found in {JSON_DIR}.")

    existing_ids = set(vector_store.get().get("ids", []))
    pending_documents = [
        (index, document)
        for index, document in enumerate(documents)
        if f"question-{index}" not in existing_ids
    ]
    if not pending_documents:
        return 0

    batch_size = 10
    for batch_start in range(0, len(pending_documents), batch_size):
        pending_batch = pending_documents[
            batch_start : batch_start + batch_size
        ]
        batch = [document for _, document in pending_batch]
        batch_ids = [f"question-{index}" for index, _ in pending_batch]
        for attempt in range(3):
            try:
                vector_store.add_documents(batch, ids=batch_ids)
                break
            except Exception as error:
                if "429" not in str(error) or attempt == 2:
                    raise
                time.sleep(45)
        if batch_start + batch_size < len(documents):
            time.sleep(2)
    return len(documents)
