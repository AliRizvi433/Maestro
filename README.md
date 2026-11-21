# Maestro

Maestro is an AI-powered exam preparation assistant for KIIT University students. It uses Google Gemini for answer generation, local sentence embeddings by default for retrieval, LangChain for RAG orchestration, and ChromaDB for vector search.

## Features

- Ask questions from past exam papers.
- Filter retrieval by subject, year, exam type, semester, and other paper metadata.
- Find repeated questions using semantic similarity across exam years.
- Generate mock papers from actual historical questions.
- Answer, explain, and prepare study material using retrieved paper context.
- Show the past-paper sources used by a RAG answer.

## Data

Historical papers are stored in `json_subs/`:

- `AI.json`
- `CC.json`
- `ML.json`
- `UHV.json`

The vector index extracts supported question text and metadata from these files.

Current data status:

- **AI:** Structured question parts and aggregate question text are available.
- **ML:** Structured sub-questions and section questions are available.
- **UHV:** Only exam instructions and constraints are available in the current JSON file, so it has limited mock-paper coverage.
- **CC:** `CC.json` currently contains Artificial Intelligence records rather than Cloud Computing records. Those mismatched records are excluded by the subject guard, so CC does not display AI questions. Replace `CC.json` with genuine Cloud Computing exam data to enable CC retrieval and repeated questions.

## Technology

- Google Gemini generation: `gemini-3.5-flash-lite`
- Embeddings: local `all-MiniLM-L6-v2` model by default
- LangChain
- ChromaDB
- Python virtual environment

## Setup

Run commands from the project root in Windows PowerShell.

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_BACKEND=local
```

Set `EMBEDDING_BACKEND=gemini` to use `gemini-embedding-001` instead. Gemini embedding access is subject to API quota limits.

Keep the API key private. `.env` is ignored by Git.

## Build the vector index

```powershell
python build_index.py
```

The index is stored in `chroma_question_db/`. If a build is interrupted, rerun `python build_index.py`; it resumes from missing question IDs.

## Run Maestro

```powershell
python main.py
```

Start the Streamlit interface:

```powershell
streamlit run app.py
```

The interface provides subject and exam-filter controls, chat, repeated-question lookup, sample-paper generation, and retrieved-source context.

Select one of `AI`, `CC`, `ML`, or `UHV`, then enter an optional year and exam type. At the question prompt:

- Enter an exam question for a Gemini-generated RAG answer.
- Enter `sample paper` to generate a mock paper.
- Enter `repeated` or `frequently asked` to show repeated questions.
- Enter `back` to change subject.
- Enter `q` to quit.
