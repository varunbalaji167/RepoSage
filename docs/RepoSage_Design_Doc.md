# RepoSage — Design Doc
 
## Problem Statement
New contributors and new hires spend disproportionate time understanding *why* a codebase is shaped the way it is — that context lives in commit messages and PR discussions, not just in the code itself. RepoSage is a query system for a specific codebase that answers "what does this do," "what would I need to touch to change this," and "is this safe to change," by combining semantic search over code with a real dependency graph, and (in later phases) validating its own suggestions against that graph before answering.
 
## Goals
- **Phase 1 (this build):** a working prototype answering two query types — code explanation and change-risk assessment — over a real open-source repo (`encode/httpx`), backed by a genuine import dependency graph and a RAG pipeline.
- **Long-term vision:** four query intents, PR/commit-history mining for "why was this built this way" questions, and a self-correcting validation loop that checks LLM-suggested file changes against the actual dependency graph.
## Non-Goals (Explicitly Out of Scope — Phase 1)
- Design-rationale queries (requires commit/PR history mining — deferred)
- Impact-analysis queries with validation loop (deferred to Phase 2)
- Multi-repo support, webhook-triggered re-indexing
- Reranking, hybrid multi-collection retrieval
- Streaming responses, auth, deployment/scaling
- Full LangSmith eval/CI gating (basic tracing only in Phase 1)
## Target Repo & Data Source
`encode/httpx` — a moderate-size (~20k LOC), pure-Python, actively maintained HTTP client library. Phase 1 scopes to a connected 9-file subset (`_client.py`, `_models.py`, `_api.py`, `_config.py`, `_auth.py`, `_urls.py`, `_exceptions.py`, `_types.py`, `_utils.py`, ~4,500 LOC) chosen from real import relationships, not arbitrarily.
 
## Proposed Architecture (Phase 1)
 
```
                 FastAPI (/ask)
                      │
                 classify (LLM + Pydantic structured output)
                      │
        ┌─────────────┴─────────────┐
   code_explanation           risk_assessment
   (LCEL RAG chain:            (pure algorithm:
   retriever│llm│parser         import graph in/out-degree
   over ChromaDB                + git log change frequency
   code_chunks collection)      — no LLM call needed)
```
 
- **ChromaDB:** single `code_chunks` collection, file-level chunking, metadata = file path
- **Dependency graph:** built via Python `ast` module + `networkx.DiGraph`
- **LangGraph:** `StateGraph` with one conditional router splitting into the two intent branches
- **Observability:** manual logging in Phase 1; LangSmith tracing added once nodes are stable
## Future Improvements (Phase 2+)
- Mine PR review comments + commit messages (GitHub GraphQL API) into a second ChromaDB collection to answer "why was this designed this way"
- Add an `impact_analysis` node + validation loop: LLM proposes affected files, a validator node cross-checks the proposal against the dependency graph and loops back on mismatch
- Reranking and metadata-filtered hybrid retrieval across collections
- LangSmith eval dataset + CI-gated prompt regression testing
- Multi-repo support with incremental, webhook-triggered re-indexing
## Success Criteria (Phase 1)
A user can ask "what does `Client.send` do?" and get a grounded, cited explanation from real code, and ask "how risky is it to change `_exceptions.py`?" and get a structured, graph-backed risk score — both served through a single FastAPI endpoint.
 
