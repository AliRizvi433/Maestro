import json
from pathlib import Path

import streamlit as st

from analysis import get_top_repeated_questions
from main import chain, get_response_text
from paper_generator import generate_sample_paper
from vector import vector_store


ROOT_DIR = Path(__file__).resolve().parent
JSON_DIR = ROOT_DIR / "json_subs"
SUBJECTS = ["AI", "ML", "CC", "UHV"]


@st.cache_data
def get_filter_options(subject):
    file_path = JSON_DIR / f"{subject}.json"
    if not file_path.exists():
        return [], []

    with file_path.open(encoding="utf-8") as file:
        exams = json.load(file)

    years = sorted(
        {str(exam["year"]) for exam in exams if exam.get("year") is not None},
        key=lambda value: int(value) if value.isdigit() else value,
    )
    exam_types = sorted(
        {str(exam["exam_type"]) for exam in exams if exam.get("exam_type")}
    )
    return years, exam_types


def build_retriever(subject, year, exam_type):
    conditions = [{"subject": subject}]
    if year:
        conditions.append({"year": year})
    if exam_type:
        conditions.append({"exam_type": exam_type})

    where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}
    return vector_store.as_retriever(
        search_kwargs={
            "k": 6,
            "filter": where_filter,
        }
    )


st.set_page_config(
    page_title="Maestro Exam Assistant",
    page_icon="M",
    layout="wide",
)

st.title("Maestro")
st.caption("KIIT exam preparation assistant")

if not vector_store.get().get("ids"):
    st.error("The vector index is empty. Run `python build_index.py` before using the UI.")
    st.stop()

with st.sidebar:
    st.header("Exam filters")
    subject = st.selectbox("Subject", SUBJECTS)
    years, exam_types = get_filter_options(subject)
    year = st.selectbox("Year", ["All years", *years])
    exam_type = st.selectbox("Exam type", ["All exam types", *exam_types])

    year_filter = None if year == "All years" else year
    exam_type_filter = (
        None if exam_type == "All exam types" else exam_type
    )

    st.divider()
    if st.button("Find repeated questions", use_container_width=True):
        repeated = get_top_repeated_questions(subject)
        if repeated:
            st.session_state.pending_action = (
                "assistant",
                "## Repeated questions\n\n"
                + "\n".join(f"{index}. {question}" for index, question in enumerate(repeated, 1)),
            )
        else:
            st.session_state.pending_action = (
                "assistant",
                "No repeated questions were found for this subject.",
            )

    if st.button("Generate sample paper", use_container_width=True):
        paper = generate_sample_paper(
            subject,
            year_filter,
            exam_type_filter,
        )
        st.session_state.pending_action = ("assistant", paper)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

pending_action = st.session_state.pop("pending_action", None)
if pending_action:
    role, content = pending_action
    st.session_state.messages.append({"role": role, "content": content})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question from the past papers")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    lowered_question = question.lower()
    if "sample paper" in lowered_question:
        answer = generate_sample_paper(
            subject,
            year_filter,
            exam_type_filter,
        )
    elif "repeated" in lowered_question or "frequently asked" in lowered_question:
        repeated = get_top_repeated_questions(subject)
        answer = (
            "## Repeated questions\n\n"
            + "\n".join(
                f"{index}. {item}" for index, item in enumerate(repeated, 1)
            )
            if repeated
            else "No repeated questions were found for this subject."
        )
    else:
        retriever = build_retriever(subject, year_filter, exam_type_filter)
        try:
            retrieved_docs = retriever.invoke(question)
            exam_paper = "\n\n".join(
                f"[{doc.metadata.get('year')}, {doc.metadata.get('exam_type')}] "
                f"{doc.page_content}"
                for doc in retrieved_docs
            )
            result = chain.invoke(
                {
                    "exam_paper": exam_paper,
                    "question": question,
                }
            )
            answer = get_response_text(result)
            if retrieved_docs:
                answer += "\n\n### Retrieved context"
                for index, doc in enumerate(retrieved_docs, 1):
                    answer += (
                        f"\n\n**Source {index}:** "
                        f"{doc.metadata.get('year')}, "
                        f"{doc.metadata.get('exam_type')}\n\n"
                        f"{doc.page_content}"
                    )
        except Exception as error:
            answer = f"Unable to answer this request: {error}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
