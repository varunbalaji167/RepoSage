# RepoSage — Design Doc

**Status: Phase 1 complete.** Both query types work end-to-end through a
single API endpoint, backed by real dependency and call graphs, tested
(94 automated tests + an opt-in real end-to-end test), traced (LangSmith),
containerized, and deployed. For the full, step-by-step log of every
design decision and real bug found while building this, see
[`DECISIONS.md`](./DECISIONS.md).

## Problem Statement

New contributors and new hires spend disproportionate time understanding
*why* a codebase is shaped the way it is — that context lives in commit
messages and PR discussions, not in the code itself. RepoSage is a query
system for a specific codebase that answers "what does this do," "what
would I need to touch to change this," and "is this safe to change," by
combining semantic search over code with a real dependency graph.

## Goals

- **Phase 1 (complete):** two query intents — `code_explanation` and
  `risk_assessment` — served over a real open-source repo
  (`encode/httpx`), backed by a genuine import dependency graph, a
  function-level call graph, and a RAG pipeline, all reachable through a
  single FastAPI endpoint.
- **Long-term vision:** design-rationale queries via PR/commit-history
  mining, and a self-correcting validation loop that checks LLM-suggested
  changes against the actual dependency graph. See **Phase 2** below.

## Target Repo & Data Source

`encode/httpx` — a moderate-size (~20k LOC), pure-Python, actively
maintained HTTP client library. Phase 1 scopes to a connected 9-file
subset (`_client.py`, `_models.py`, `_api.py`, `_config.py`, `_auth.py`,
`_urls.py`, `_exceptions.py`, `_types.py`, `_utils.py`, ~4,500 LOC),
chosen from real import relationships, not arbitrarily.

## Architecture (as built)

```
                          Browser UI (app/static/)
                                   │
                         FastAPI  POST /ask
                       (app/main.py, app/api/)
                                   │
                        LangGraph orchestration
                       (app/orchestration/graph.py)
                                   │
                    classify_node (structured-output
                    intent classifier + no-LLM filename/
                    class-name extraction)
                                   │
                         route_by_intent
              ┌────────────────────┴────────────────────┐
       code_explanation_node                    risk_assessment_node
      (LCEL RAG chain: retriever                (filters compute_risk_scores()
       │ 1-hop call-graph expansion              to the requested file(s);
       │ prompt │ llm │ parser,                  asks for clarification if
       over ChromaDB code_chunks)                 no file was identified)
              └────────────────────┬────────────────────┘
                          format_response_node
                      (normalizes both branches into
                       one QueryResponse schema)
```

**Indexing** (`app/indexing/`) — AST-based chunking at function/class
granularity, with a sliding-window fallback for oversized functions;
embedded with `jina-embeddings-v2-base-code` (local, free) into a
persistent ChromaDB collection.

**Graph analysis** (`app/graph_analysis/`) — a file-level import graph
(`networkx.DiGraph`), a separate function-level call graph using **Class
Hierarchy Analysis** for `self.method()` resolution (never guesses a
single receiver when a call could dispatch to any subclass override),
and risk scoring combining blast radius (graph ancestors) with commit
frequency (`git log`), gated (not weighted-summed): high impact + low
churn is what actually earns HIGH risk.

**Retrieval** (`app/chains/code_explanation_chain.py`) — overfetches raw
chunks and deduplicates by *function identity* (not chunk identity)
before applying the top-k cutoff, so a function split across multiple
chunks doesn't waste retrieval slots competing against itself. A small,
narrowly-scoped exact-identifier boost guarantees inclusion when the
question literally names a `Class.method` that pure semantic similarity
ranked below the cutoff — a real limitation found via live testing (see
`DECISIONS.md`, Step 9), not a hypothetical.

**Classification** (`app/classification/`) — structured LLM output
(`Literal["code_explanation", "risk_assessment"]`) for intent, kept
deliberately separate from filename/class-name extraction, which uses no
LLM at all (a closed, 9-file vocabulary doesn't need one).

**Error handling** (`app/errors.py`) — a three-tier taxonomy: raw
library exceptions are wrapped at the module boundary where they occur;
a small set of common app-wide types (`ServiceUnavailableError`,
`InternalError`, ...) each carry a fixed HTTP status and a `retryable`
flag; narrow domain subclasses (`RetrievalError`, `RiskDataError`,
`CallGraphError`, `LLMUnavailableError`) add context where needed. LLM
calls get per-error-type handling: rate limits fall back to a lighter
model, transient errors get exponential backoff, misconfigurations
(bad key, wrong model name) fail immediately rather than retry a real bug.

**Testing** — 94 fast, mocked tests (no API key or real index needed)
run in CI on every push, plus an opt-in real end-to-end test
(`RUN_E2E=1`) against the live API, index, and call graph — the only
thing that can catch a wrong assumption about a real dependency's
behavior, which mocks structurally cannot.

**Observability** — LangSmith tracing (env-var only, no code
instrumentation needed — every LCEL `Runnable` and LangGraph node already
runs through the callback-manager system tracing plugs into).

**Deployment** — Dockerized; the vector index and target repo clone are
baked into the image at build time (the app never writes to either at
runtime), so every deployed container is reproducible with no
first-request indexing latency.

## Key design decisions worth knowing before extending this

- **Precision over recall, everywhere a wrong answer could be worse than
  a missing one.** Unresolvable calls in the call graph are logged, never
  guessed. Citations are built from retrieval metadata after the LLM
  runs, never trusted from the LLM's own text.
- **Every cross-module data shape is a real Pydantic schema**
  (`RiskScore`, `QueryClassification`, `QueryResponse`,
  `CodeExplanationResult`, `CodeChunk`), not a loose dict — checked
  against a deliberate test: does this data cross a real module boundary,
  or is it purely internal scratch state? Only the former gets a schema.
- **Resource construction is lazy and cached**, never eager at import
  time — the entire app can be imported with zero side effects (no API
  key, no network, no model download required just to import a module).
- **Retrieval hybrid-reranking was deliberately deferred, not built.**
  The exact-identifier boost is a narrow, targeted patch for one
  confirmed failure mode, not a general solution — see Phase 2.

## Known Limitations

- No type inference in the call graph — calls on variables of unknown
  type (`response.json()`) are unresolvable by design; fixing this would
  mean reimplementing a slice of what `mypy`/`Jedi` already do.
- Retrieval uses a narrow exact-identifier boost, not full hybrid
  dense+lexical retrieval or reranking.
- Single-repo only; no incremental/webhook-triggered re-indexing.
- `vector_store.py` has no automated tests — its logic is entangled with
  the embedding model download.
- No auth, no streaming responses, no multi-tenant scaling.

## Success Criteria (Phase 1) — met

A user can ask *"what does `Client.send` do?"* and get a grounded, cited
explanation from real code, and ask *"how risky is it to change
`_exceptions.py`?"* and get a structured, graph-backed risk score — both
served through a single FastAPI endpoint, verified against the real
system, not just a mocked test suite.

---

## Phase 2 — Future Improvements

- **Design-rationale queries.** Mine PR review comments and commit
  messages (GitHub GraphQL API) into a second ChromaDB collection, to
  answer "why was this designed this way" — a genuinely different kind
  of question from what Phase 1 answers.
- **Impact-analysis validation loop.** An `impact_analysis` node where
  the LLM proposes affected files for a change, and a validator node
  cross-checks that proposal against the real dependency graph, looping
  back on mismatch rather than trusting the LLM's guess.
- **Proper hybrid retrieval and reranking.** Phase 1's exact-identifier
  boost is a narrow patch for one confirmed failure mode (near-identical
  sibling methods, like sync/async twins, losing to each other in pure
  semantic search). A real fix means genuine hybrid dense+lexical
  retrieval (e.g. BM25 alongside embeddings) or a dedicated reranking
  step — now empirically justified by a real production bug, not
  speculative.
- **Multi-repo support** with incremental, webhook-triggered
  re-indexing, instead of a build-time-baked, single-snapshot index.
- **Type-inference-aware call resolution**, closing the
  `response.json()`-style gap — a substantial undertaking (real type
  inference, not a quick addition), so only worth it if the current
  limitation proves costly in practice.
- **A real eval dataset + CI-gated prompt regression testing** via
  LangSmith, rather than the current opt-in single-run E2E test —
  meaningful once there's enough real usage to build a representative
  eval set from.
- **Auth, streaming responses, and multi-tenant deployment scaling** —
  all deliberately out of scope until there's a real multi-user need,
  not built speculatively ahead of it.