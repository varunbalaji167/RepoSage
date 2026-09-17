# RepoSage

A query system for the `encode/httpx` codebase. Ask what a piece of code
does, or how risky it is to change — answered with real citations and
graph-backed risk scoring, not guesses.

See `RepoSage_Design_Doc.md` for the full architecture rationale and
`DECISIONS.md` for a step-by-step log of every real design decision and
bug found while building this.

## Setup

**1. Clone this repo, then set up the target repo it analyzes:**
```bash
git clone <your-fork-url>
cd RepoSage
git clone https://github.com/encode/httpx.git target_repos/httpx
```
Must be a **full clone**, not shallow — risk scoring relies on complete
commit history via `git log`.

**2. Create and activate a Python environment (3.11+), then install:**
```bash
pip install -r requirements.txt
```

**3. Set up your `.env` file** (never commit this — see `.gitignore`):
```
GOOGLE_API_KEY=your-gemini-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key-here
LANGCHAIN_PROJECT=RepoSage
```
Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com),
a free LangSmith key at [smith.langchain.com](https://smith.langchain.com).
LangSmith vars are optional — tracing simply won't activate without them.

**4. Build the vector index** (one-time; downloads the Jina embedding
model on first run):
```bash
python -m app.indexing.vector_store
```

## Running

```bash
uvicorn app.main:app --reload
```
Open `http://localhost:8000` for the UI, or `http://localhost:8000/docs`
for the interactive API docs.

## Testing

```bash
pytest tests/ -v
```
Fast, mocked, no API key needed — this is what runs in CI.

For a genuine end-to-end test against the real API and real index (costs
real LLM quota):
```bash
RUN_E2E=1 pytest tests/test_e2e.py -v
```
**This E2E test is not run in CI** — it needs a live `GOOGLE_API_KEY` and
costs real quota on every run, so it's left as a manual, local-only check
before a release rather than something that runs on every push/PR.

## Architecture, briefly

```
question -> classify_node -> route_by_intent -> code_explanation_node
                                               -> risk_assessment_node
                                               -> format_response_node -> QueryResponse
```

- `app/indexing/` — AST-based code chunking + Chroma vector index
- `app/graph_analysis/` — import graph, function-level call graph (with
  Class Hierarchy Analysis), risk scoring
- `app/chains/` — the LCEL RAG chain for code explanation
- `app/classification/` — structured-output intent classifier
- `app/orchestration/` — the LangGraph wiring everything together
- `app/api/`, `app/main.py` — the FastAPI layer (no business logic)
- `app/errors.py` — the shared error taxonomy (library errors wrapped at
  their boundary → common app-wide types → HTTP status via a global
  exception handler)

Full rationale for every one of these choices — including the real bugs
found and fixed along the way — is in `DECISIONS.md`.

## Known limitations (deliberately out of scope for this phase)

- No type inference in the call graph (calls on variables of unknown
  type are logged as unresolved, never guessed)
- Retrieval uses a narrow exact-identifier boost, not full hybrid
  dense+lexical reranking — see `DECISIONS.md`'s Step 9 entry
- Single-repo, no incremental re-indexing, no auth, no streaming
- `vector_store.py` has no automated tests (entangled with the embedding
  model download — a reasonable follow-up)