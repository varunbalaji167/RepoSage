# Design Decisions

This document records the key technical decisions made while building this project, along with the reasoning, trade-offs, and real bugs found along the way. It's meant as a reference for *why* things are the way they are — not just *what* they are.

---

## Step 1 — Environment & Repo Setup

**Decision: Full git clone, not a shallow clone (`--depth 1`)**
- The risk-scoring metric in Step 3 needs real commit history per file (`git log`), which a shallow clone doesn't provide.
- The repo was only 13MB, so the cost of a full clone was negligible.
- Lesson: a shallow clone would have looked fine through Steps 1–2 and only broken later — worth checking what *future* steps need before locking in early setup choices.

---

## Step 2 — AST Parsing & Dependency Graph

**Decision: edges represent "A imports B"**
- This lets `nx.ancestors()` on a file directly answer "what depends on this file" — the blast-radius question Step 3's risk score needs.

**Decision: exclude `if TYPE_CHECKING:` imports from the graph**
- These never execute at runtime and can't cause a real break. Only imports Python actually runs should count toward risk.
- This matched the design doc's prediction that `_types`, `_exceptions`, and `_utils` would have zero outgoing edges — confirmed.

**Bug: recursive import detection missed nested imports**
- The first version checked if a node "was" an import via `iter_child_nodes`, which worked at the top level but skipped imports inside an `else` branch (since those are passed in as the node itself, not as someone else's child).
- Fixed by rewriting the walker to check every statement directly and consistently, regardless of nesting (module body, if-body, function body).

**Known limitation: no true circular-import detection**
- The graph builder only parses text; it never executes or loads files the way Python's import system does. So it can't detect/crash on real circular imports — it just records both edges of the cycle and moves on.

---

## Step 3 — Risk Scoring

**Decision: impact × likelihood matrix, not a single blended formula**
- Impact = blast radius (`nx.ancestors`), likelihood = commit frequency.
- More explainable than a blended score, and avoids arbitrary weighting between two signals that aren't correlated in the data (e.g. `_client` had the most commits but the smallest blast radius).

**Finding: commit count alone is a misleading risk signal**
- `_client`: 145 commits (highest), but only 1 dependent → correctly ranked LOW risk.
- `_types`: moderate commit count, but highest blast radius → ranked HIGH.

**Design detail: impact is a gate, not an equal input**
- The if/elif structure means churn only affects the outcome once impact already qualifies as high.
- This makes the score robust to commit-count changes on low-impact files (verified by tracing what happens if `_client`'s commit count changed).

**Decision: `statistics.median()` from live data, not hardcoded thresholds**
- Correct for the current curated 9-file connected subgraph.
- **Flagged as a Phase 2 concern:** a median split likely breaks down on a larger, more skewed file set (e.g. the full ~40-file httpx repo) — percentile-based thresholds would probably be needed instead.

---

## Step 4 — Chunking & Embedding (ChromaDB Indexing)

**Decision: chunk at function/class level via AST, not whole-file**
- Whole-file chunking would require manually judging "is this file big or small?" for every file — a judgment that has to be redone by hand as files are added.
- Function/class-level chunking makes that decision automatically, node by node, so the same logic scales from 9 files to 900.
- Oversized single functions/classes fall back to a sliding-window split so no chunk is too large to embed meaningfully.

**Decision: embed with `jina-embeddings-v2-base-code` (local, free), not a general text model**
- A general text embedder anchors on surface wording (variable names, docstring phrasing) rather than behavior — two functions that do the same thing but are named/documented differently can end up far apart in vector space.
- A code-trained model avoids that gap.

**Investigated: could same-named methods (e.g. a `@property` getter/setter pair) collide in Chroma?**
- Chunk IDs are built from file path + name + start line + index.
- Confirmed via real output that `start_line` always differs between them (Python statements can't share a line) — no collision occurs. `start_line` was added for debugging traceability, not collision-prevention, but it solved this problem as a side effect.

**Open gap (deferred to Step 5):** functions split into multiple parts (`[part 1/2]`, `[part 2/2]`) aren't reassembled at retrieval time. If a query matches only one part, the LLM could confidently answer using incomplete logic.

---

## Step 5a — Function-Level Call Graph (for Retrieval Expansion)

**Decision: a second graph, separate from Step 2's file-level import graph**
- Nodes are individual functions/methods; edges mean "A calls B."
- Purpose: the retriever expands beyond pure semantic match — if the top-matched chunk is `Client.send`, also pull in what it directly calls (1 hop), since a method's full behavior often isn't visible from its own text alone.

**Decision: resolve `self.method()` via Class Hierarchy Analysis (CHA)**
- Since the caller's actual runtime type isn't tracked, add an edge to *every* possible receiver (the class itself + all subclasses that override the method) — not one guessed class.
- Rationale: a single wrong guess would silently corrupt LLM context later; a slightly wider but correct edge set doesn't.
- `super().method()` is the opposite case — a fixed lookup starting at the parent class, since `super()` unambiguously means "my parent."

**Bug: two call patterns initially missed**
- Bare constructor calls (`URL(...)`, `Headers(...)` → resolving to that class's `__init__`), and `super().__init__(...)` itself.
- Caught by grouping 726 unresolved calls by name and noticing `URL`/`Headers` weren't builtins.

**Known, permanent limitations (inherent to static analysis without a type checker):**
1. Calls on variables of unknown type (e.g. `response.json()`) are unresolvable — would need real type inference (mypy/Jedi-level tooling), out of scope.
2. If two target files define same-named top-level functions, cross-file resolution could silently pick the wrong one (first match in import order). Didn't occur in httpx's actual 9 files, but the resolver doesn't guard against it — would need per-name import tracking to fix properly.

**Guiding principle:** precision over recall — skip-and-log rather than guess. For a graph feeding LLM context, a missing edge is a safe gap; a wrong edge is invisible corruption.

---

## Step 5 — LCEL `code_explanation` Chain

**Decision: retrieve `k=3` chunks**
- Chunks are already function/class-level (not tiny fragments), so 3 gives enough room to catch a related chunk without flooding the LLM's context with diluting noise.

**Decision: reassemble split chunks by `file_path` + `parent_class` + base name**
- Not name alone — two different split methods in the same file could collide on name alone.
- Otherwise the LLM would see only one part of a method and could confidently describe incomplete logic as if it were the whole thing.

**Decision: attach citations after the LLM call, not before**
- Once text passes through the LLM and its output parser, the original `Document` objects — and their file/line metadata — no longer exist; only raw generated text remains.
- `RunnableParallel` carries the retrieved documents alongside the LLM's answer so metadata survives to where citations are built.

**Bug: Chroma's `where` clause needed explicit `$and` wrapping**
- This version of `chromadb` requires exactly one top-level operator per `where` clause, not a flat dict with two keys.

**Decision: generation LLM = Google Gemini (free tier via AI Studio)**

**Bug: expensive work was running at import time**
- The whole chain module built the Chroma collection, embedding model, call graph, and LLM client all at *import* time.
- Fixed with lazy, cached (`lru_cache`) resource getters, passing dependencies explicitly into pure logic functions instead of reading module globals.
- This mattered for testing: importing the module previously required a live API key, network access, and a downloaded model just to import it.

**Note: Gemini 3.6+ deprecated the `temperature` parameter entirely**
- Not just ignored — documented to return an HTTP 400 error in future model generations if still passed.
- Removed it; relying on structured output schema + explicit system instructions for determinism, per Google's migration guidance.

---

## Step 6 — Structured-Output Intent Classifier

**Decision: narrow output schema — just `intent` (`Literal["code_explanation", "risk_assessment"]`)**
- Bundling filename/function extraction into the same LLM call would mean one validation failure couldn't tell us *which part* broke.
- Keeping classification and extraction separate means each can be tested and debugged independently.

**Decision: `extract_target_files` uses no LLM — plain string matching**
- The set of known filenames (`TARGET_FILES`) is small, fixed, and closed — deterministic string matching is free, instant, and can never hallucinate a nonexistent file. LLM calls are reserved for genuinely open-ended judgment.

**Bug: extraction could only return one file, non-deterministically**
- The original function returned only one file even when a question mentioned multiple (e.g. "does `_client.py` depend on `_models.py`?").
- *Which* one it returned was non-deterministic across runs, because it iterated over a Python `set`, and set iteration order isn't guaranteed stable for equal-length strings.
- Fixed by returning a list of all matches, ordered by first appearance in the question text.

**Bug: class names (e.g. `Client`) couldn't be resolved to their file**
- Only literal filename mentions worked.
- Fixed by adding `build_class_to_file_map()` in `call_graph.py`, reusing only the first (class-registration) pass of the existing AST-walking logic — deliberately skipping the expensive second pass (full call resolution), since this doesn't need the whole 280-node call graph.

**Decision: require `.py` suffix when the underscore is dropped (e.g. `client.py` vs `_client.py`)**
- People naturally drop the leading underscore in prose, but a bare word like `types` would otherwise false-positive-match ordinary sentences (e.g. "what types of arguments does `send` take?").
- The `.py` suffix buys back the confidence lost from dropping the underscore, without needing a per-file exception list.

**Operational note (not a code fix):** Gemini's `google-genai` SDK warns that automatic function calling via `Models.generate_content` (the pattern `langchain_google_genai` currently uses) is being deprecated in favor of the `Chats` module. Logged as an external dependency risk to watch — not something to engineer around now.

---

## Step 7 — LangGraph Orchestration (Router + 2 Intents)

**Decision: nodes return partial state updates, not full state**
- LangGraph merges each node's return into running state using per-field logic.
- Partial updates mean each node only needs to know its own fields — e.g. `code_explanation_node` doesn't need to know about `target_files`.

**Decision: `route_by_intent` is separate from `classify_node`**
- Its job is different: translating `state["intent"]` into a literal graph node name (a wiring detail) vs. making a classification decision.
- Keeps `classify_query` and `extract_target_files` reusable and testable independent of graph topology.

**Decision: `format_response_node` is its own graph node, not deferred to the FastAPI layer**
- Normalizing two structurally different result shapes into one `final_output` is still a business-logic decision, not a presentation concern.
- Matches `main.py`'s own convention that the API layer "contains no business logic."

**Bug: duplicated risk-scoring cache**
- `compute_risk_scores()` (which runs `git log` per file) had a private, separately-cached copy inside `code_explanation_chain.py`.
- Left alone, `risk_assessment_node` would have built a second, independent cache of the same data.
- Fixed by moving it to one shared `get_risk_by_file_stem()` in `risk_scoring.py`, imported by both modules — `git log` now runs once per process.

**Decision: promote key dict/dataclass types to Pydantic schemas**
- `final_output`, `code_explanation_result`, and `CodeChunk` genuinely cross module boundaries, same category as `RiskScore`/`QueryClassification`, which already had proper schemas.
- Fixed by adding `QueryResponse` and `CodeExplanationResult`, and moving `CodeChunk` into `schemas.py`.
- Deliberately did **not** convert `UnresolvedCall` the same way — verified it never leaves `call_graph.py`, so converting it would be uniformity for its own sake.

**Bug: silent drop of unrecognized target files**
- When `risk_assessment_node` received an unrecognized target file alongside a valid one, it used to filter it out with no explanation.
- Fixed: (a) if *none* of the requested files are recognized, ask for clarification with a list of valid filenames; (b) if *some* are recognized, answer for those and explicitly say which weren't understood.

---

## Step 8 — FastAPI Wrapper, Frontend, and Error Handling

**Decision: single endpoint (`POST /ask`), per the design doc**
- No separate endpoints per intent — the graph's own classifier decides the path internally, matching the design doc's success criteria.

**Decision: plain `def`, not `async def`, for the route**
- `graph.invoke()` is synchronous and does blocking I/O (Gemini API calls, `git log` subprocess calls) — it never uses LangGraph's async path.
- An `async def` route calling it directly would block FastAPI's entire event loop per request.
- A plain `def` route runs in FastAPI's background threadpool automatically, so one slow request doesn't stall others.

**Decision: graph built once at startup (lifespan), injected via `get_graph`, not a bare module-level global**
- `build_graph()` itself is cheap — the expensive resources (Chroma, call graph, LLM client) are already lazy-cached and only trigger on first real `.invoke()`.
- Using a FastAPI dependency instead of a global lets tests swap in a fake graph via `dependency_overrides`, with no real API key or network access needed.

**Decision: `/ask` registered before the catch-all static file mount**
- `StaticFiles(html=True)` serves `index.html` for any unmatched path if mounted first — including `/ask` — which would shadow the route entirely.
- Verified directly with `TestClient`, not assumed.

**Bug: retrieval failure found through live browser testing, not the test suite**
- "What does `Client.request` do?" returned "not included in context" even though the method exists.
- Root cause: the sliding-window chunk-splitting feature (Step 4) and top-k retrieval (Step 5) each tested fine in isolation but had never been stress-tested together — a split function could win multiple top-k slots against itself, crowding out genuinely relevant, distinct functions.
- Fixed by overfetching raw chunks and deduplicating by *function* identity (not chunk identity) before applying the top-k cutoff.

**Decision: deliberate, non-generic frontend design**
- Built for a developer-tool audience: cool neutral palette, one confident accent, IBM Plex Sans/Mono, single-column layout, no chat-bubble metaphor (single-query tool, not a conversation).
- Checked against common AI-generated-UI tells (cream+terracotta, near-black+neon, card-shadow kits) before building, not after.

**Decision: three-tier error taxonomy, not ad-hoc try/except**
1. Raw library exceptions (`chromadb`, `subprocess`, `langchain_google_genai`) are caught and wrapped into custom types at the exact point of the library call — never leaked across module boundaries.
2. Common, reusable app-wide types (`ServiceUnavailableError`, `InternalError`, `ValidationError`, `RateLimitedError`), each with a fixed HTTP status and a `retryable` flag.
3. Narrow domain subclasses (`RetrievalError`, `RiskDataError`, `CallGraphError`, `LLMUnavailableError`) for cases needing extra context, each subclassing a common type so generic handling still works via `isinstance`.
- Deliberately HTTP-agnostic — lives in the graph/chain layer, which is also invoked directly by scripts, not just the API.
- Only `main.py`'s global exception handlers map error type to HTTP status (not per-route try/except) — the more correct, DRY FastAPI pattern, letting `routes.py` shed its own try/except entirely.

**Decision: retry/fallback strategy chosen per error type**
- Misconfigurations (bad API key, wrong model name) fail immediately as `InternalError` (not retryable) — retrying would waste time and mask a real bug.
- Genuinely transient failures (`ServerError`) get exponential backoff on the same model.
- Rate-limit/quota errors skip retry entirely and fall back to a lighter model (`gemini-3.1-flash-lite`) for that one call — stateless, no cooldown tracking needed, since the next request just tries primary again fresh.

**Bugs found while building this step:**
1. `subprocess.run()` doesn't raise on a non-zero exit code by default — a genuinely failed `git log` was silently reported as "0 commits" instead of surfacing as an error, risking miscategorized low-churn files.
2. The original API error handling hid the real exception from both the client (intentional) *and* the server logs (not intentional) — failures were invisible even during debugging.

---

## Step 9 — End-to-End Test + LangSmith Trace

**Finding: tracing needed zero code changes**
- Setting `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` in `.env` was sufficient — every LCEL `Runnable` and LangGraph node runs through the same callback-manager system LangSmith's tracer plugs into.
- Confirmed via a real trace: `LangGraph` at the root, expanding into `classify` → `route_by_intent` → `code_explanation_node`, with the inner LCEL chain unfolding one level deeper inside that node.

**The E2E test justified itself immediately**
- All 94 mocked tests were green, but a real question through the real browser ("What does `Client.request` do?") returned "not included in context" — a genuine live bug no amount of mocking could catch, because it was a wrong assumption about real embedding rankings, not a logic bug.

**Bug had two layers, found only by tracing real data:**
1. **Stopping-logic error:** the overfetch + dedupe-by-identity fix (Step 8) broke the loop the instant it collected `TOP_K` distinct identities, rather than scanning the full overfetched window. If the first few raw ranks were already distinct, the loop stopped immediately — even though a more relevant function was one rank further down. Confirmed: `Client.request` ranked 4th, one place past where the old loop stopped, inside a 12-chunk window it never got to see.
2. **Semantic ranking issue:** even after fixing (1), pure semantic similarity ranked `AsyncClient.request` (a near-identical sibling) ahead of `Client.request` — the method literally named in the question — because embeddings don't specially privilege an exact name match over a semantically similar one.

**Fix scope kept deliberately narrow:**
- Layer 1: scan the entire overfetch window before cutting to `TOP_K`.
- Layer 2: a small, targeted exact-identifier boost — if the question contains a literal `"ClassName.method"` substring matching a candidate in the scanned window, guarantee its inclusion as one bonus context chunk (never displacing a natural top-k pick).
- Scoped to the dotted `Class.method` pattern only (case-sensitive, low false-positive risk) — not bare method names, which carry the same ambiguity risk solved earlier for `extract_target_files`.
- A full fix (hybrid dense+lexical retrieval, reranking) is explicitly named as Phase 2 / Non-Goals in the design doc — this patch solves the concrete symptom without scope creep.

**Decision: regression tests built from actual real ranking data**
- The exact observed rank order (`Client.request` at position 4) is reproduced directly in test fixtures, so this specific bug can never silently regress unnoticed.

**Decision: E2E test is opt-in, not part of the default suite**
- Costs real LLM quota and needs a live API key, so `pytest tests/` skips it by default (`RUN_E2E=1` to run deliberately).
- This is the one test that trades speed/determinism for the one guarantee no mock can give: that assumptions about a real dependency's actual behavior are correct, not just internally consistent.