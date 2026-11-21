import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from analysis import get_top_repeated_questions
from paper_generator import generate_sample_paper
from vector import vector_store


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT_DIR, ".env"))
subjects = ["AI", "ML", "CC", "UHV"]

model = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    temperature=0.2,
    api_key=os.environ.get("GEMINI_API_KEY"),
)
prompt = PromptTemplate.from_template(
    """
You are Maestro, an exam-preparation assistant for KIIT University.
Help the student with past KIIT exam questions and subject topics.

Use the supplied past-paper context first. You may:
- solve a question step by step;
- explain the underlying concept at a beginner-friendly level;
- provide examples and exam-oriented notes;
- find and return related questions from the context;
- generate additional practice questions similar to the supplied questions;
- compare or correct concepts when the student asks.

If the context is insufficient, say exactly what information is missing instead of
inventing an exam answer. Clearly label assumptions.
If the request is unrelated to exam preparation, say:
"I cannot help with that."

Past-paper context:
{exam_paper}

Student question:
{question}
"""
)
chain = prompt | model


def get_response_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
        return "\n".join(text_parts).strip()
    return str(content)


def get_subject():
    print("Available subjects:", ", ".join(subjects))
    while True:
        subject = input("Select a subject (or 'q' to quit): ").strip().upper()
        if subject == "Q":
            return None
        if subject in subjects:
            return subject
        print("Invalid subject. Please choose from:", ", ".join(subjects))


def get_filters():
    filters = {}
    year = input("Optional year (press Enter to skip): ").strip()
    if year:
        filters["year"] = year
    exam_type = input("Optional exam type (press Enter to skip): ").strip()
    if exam_type:
        filters["exam_type"] = exam_type
    return filters


def main():
    if not vector_store.get().get("ids"):
        raise RuntimeError(
            "The vector index is empty. Run "
            "'python build_index.py' in the activated virtual environment first."
        )

    while True:
        print("-" * 88)
        subject = get_subject()
        if not subject:
            return

        filters = get_filters()
        conditions = [{"subject": subject}]
        if filters.get("year"):
            conditions.append({"year": str(filters["year"])})
        if filters.get("exam_type"):
            conditions.append({"exam_type": filters["exam_type"]})
        filter_query = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 6, "filter": filter_query}
        )

        while True:
            question = input(
                f"Ask your {subject} question ('back' changes subject, 'q' quits): "
            ).strip()
            if question.lower() == "back":
                break
            if question.lower() == "q":
                return
            if "sample paper" in question.lower():
                print(
                    "\n" + generate_sample_paper(
                        subject, filters.get("year"), filters.get("exam_type")
                    ) + "\n"
                )
                continue
            if "repeated" in question.lower() or "frequently asked" in question.lower():
                repeated = get_top_repeated_questions(subject)
                print("\nTop repeated questions:\n")
                print("\n".join(f"- {item}" for item in repeated) or "No repeated questions found.")
                continue

            retrieved_docs = retriever.invoke(question)
            exam_paper = "\n\n".join(
                f"[{doc.metadata.get('year')}, {doc.metadata.get('exam_type')}] "
                f"{doc.page_content}"
                for doc in retrieved_docs
            )
            result = chain.invoke({"exam_paper": exam_paper, "question": question})
            print(f"\nAnswer:\n{get_response_text(result)}\n")


if __name__ == "__main__":
    main()
