# Evaluation

A green test count tells you the code doesn't crash. It doesn't tell you whether the system *answers questions correctly*. This doc is about the evaluation that does. Harness and dataset: `eval/`.

## Dataset — 48 questions, written from source

6 categories, chosen so each one tests one specific behaviour.

| category | n | what it tests |
|---|---|---|
| named | 15 | the user names a function/class in the query (easy case) |
| conceptual | 15 | no identifier; retrieval has to find the right code from meaning (hard case) |
| risk | 8 | risk branch; query names 1 or 2 files |
| clarify | 2 | risk question with no file named — must ask, not guess |
| ambiguous | 2 | surface cues point the wrong way ("would break", "safe to call") |
| out of scope | 6 | answer isn't in the 9 indexed files; correct response declines |

For each question, an answer key is written down: which function in which file contains the real answer, which words the response must mention, which files a risk query is about, whether to answer/clarify/decline. **Written before running RepoSage on any of these, so labels can't have been tuned to its output.** `eval/validate_dataset.py` greps every label against the real httpx source — caught 3 labelling mistakes on first run; stays green.

"Gold function" = the entry in the answer key. For `What does Client.request do?`, the gold is `Client.request` in `_client.py` lines 771–825. A retrieved span counts as a hit if it overlaps that range.

## Procedure — 4 stages

Each writes its own file so a crash loses nothing. `--resume <dir>` picks up where the last one stopped.

1. **Deterministic (no LLM).** For each code question, grab the top-3 primary spans and the 1-hop expansion, compare to the gold by line-range overlap. In the same loop, run a naive baseline: plain `collection.query(n_results=3)` with none of RepoSage's extras (overfetch+dedupe, exact-identifier boost, split reassembly, call-graph expansion). The gap is what those extras actually buy. For risk queries, also run the filename extractor and compare to the gold target files.

2. **Pipeline (real system).** Call `graph.invoke()` on every query — real Chroma, real Gemini, no mocks. Record answer, citations, latency per node, tokens per model, retries, fallback-model use.

3. **Judge (faithfulness).** For every code answer, replay retrieval + prompt formatting to reconstruct the exact string fed to the generator. (Verified: all 31 replays reproduce identical cited spans.) Ask a second LLM to split the answer into atomic factual claims and verdict each one against that context: *supported / unsupported / contradicted*. Separately, a no-LLM sweep flags identifiers in the answer that don't appear anywhere in the httpx package.

4. **Metrics.** Rates shown as k/n with 95% Wilson score intervals — at n=48, one query moves a rate by ~2 points.

Risk gets a side check: `eval/common.py::independent_risk` recomputes each file's risk level from `ast` + `git rev-list --count` with zero imports from `app/`. Numbers compared file by file against production.

## Latest run

| | |
|---|---|
| intent routing | 48/48 |
| target-file extraction | 11/11 |
| risk — right files | 9/9 |
| risk — numbers match independent recomputation | 9/9 |
| clarification path fires | 2/2 |
| retrieval: gold reached (RepoSage) | 30/31 |
| retrieval: gold reached (naive baseline) | 28/31 |
| citations resolve to real code | 161/163 |
| answers citing the gold function | 31/31 |
| hallucination, response-level (judged) | 0/31 |
| hallucination, claim-level (judged) | 0/162 |
| abstention on out-of-scope | 3/6 |
| warm latency p50 / p95 | 2.7s / 25.3s |
| mean tokens / query | 1188 in + 168 out |
| completed without error | 48/48 |

## What the eval surfaced

Headlines aren't the useful bit; the three specific weaknesses are.

- **Abstention is weak** (3/6 out-of-scope). On half the trap questions the system writes a confident answer stitched from retrieved-but-unrelated code. Phrase-matching finds hedging in 5/6, so the model is partway there but not committing. Fix: stricter abstain prompt + a context-relevance check before generation.
- **One retrieval miss, q29.** "How does httpx handle a 307 redirect?" — primary retrieval returned `Client._send_handling_redirects` and its `AsyncClient` twin. The 3 `BaseClient._redirect_*` methods only reached context through 1-hop call-graph expansion. Sync/async duplication is still what pure semantic retrieval struggles with most.
- **One answer paraphrased away a key word, q18.** "How does httpx strip the Authorization header on a cross-origin redirect?" — retrieval found the right function (`_redirect_headers`), the answer talked around the word "Authorization". Caught by the must-mention check, not the hallucination judge.

Separately: the 25s p95 is Gemini free-tier rate-limit fallbacks to the backup model, not system latency. On the ~90% of queries where nothing is throttled, p95 is a few seconds.

## What the numbers do *not* prove

Four things worth saying out loud.

**Author bias.** Same hands built the system and the dataset. Labelling from source first prevents post-hoc tuning. It does *not* prevent unconsciously picking questions the system handles well. An external dataset — written by someone else, ideally from real usage — would stress-test this harder.

**Judge shares a brain with the generator.** Gemini judging Gemini. 0/162 is the number to doubt hardest, not the one to lead with. The two no-LLM checks are better evidence of grounding: 31/31 cite the gold function; 29/30 hit every required key fact. A cross-family judge (GPT-4, Claude) plus a sampled human audit would be a real upgrade.

**The retrieval win is category-specific.** On named queries RepoSage beats the baseline 15 vs 14. On conceptual queries, both get 14/15 — same number. The headline gap is almost entirely the exact-identifier boost paying off on the case it was built for (the real Client.request bug from Step 9). A stronger baseline (BM25 or a reranker) would compress the gap further. Cosine top-3 is deliberately a low bar.

**Risk is validated against its own spec.** The independent recomputation confirms the code implements the median-gate rule correctly. It does *not* validate the rule itself. A different formula would relabel the same files, and this check wouldn't notice.

48 queries on one repo at one commit catches regressions and compares variants well. It does not support "the system is 97% accurate" in general. The honest framing: there's now something to re-run after every change, and that run produces a specific list of weaknesses to work on next.
