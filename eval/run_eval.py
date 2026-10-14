#!/usr/bin/env python
"""Evaluation harness. See docs/EVAL.md for the procedure.

Stages: 1 deterministic (retrieval vs. gold + naive baseline), 2 pipeline (real graph),
3 judge (faithfulness), 4 report. Each writes its own file; --resume reuses finished stages.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)  # the app uses repo-relative paths (chroma_db/, target_repos/)

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(HERE / "dataset.jsonl"))
ap.add_argument("--out", default=None, help="results dir (default eval/results/<timestamp>)")
ap.add_argument("--resume", default=None, help="existing results dir: skip stages whose file exists")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--skip-judge", action="store_true")
ap.add_argument("--judge-model", default=None, help="Gemini model used as faithfulness judge (default: the generator's primary model)")
ap.add_argument("--max-retries", type=int, default=2, help="re-run a query this many times if both models are rate-limited")
ap.add_argument("--retry-wait", type=float, default=30.0)
ap.add_argument("--sleep", type=float, default=0.5, help="pause between queries")
ap.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (omit => tokens only)")
ap.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens")
ap.add_argument("--trace", action="store_true", help="keep LangSmith tracing on (default: off for eval runs)")
args = ap.parse_args()

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")
if not args.trace:  # 48+ eval runs shouldn't spam the user's LangSmith project
    for k in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        os.environ[k] = "false"

from common import (ROOT as _R, STEMS, covers, independent_risk, load_nodes,  # noqa: E402
                    pct, percentile, repo_vocabulary, tokens, wilson)

CITATION_RE = re.compile(r"^(?P<file>\S+) :: (?P<loc>.+) \(lines (?P<a>\d+)-(?P<b>\d+)\)$")
ABSTAIN_RE = re.compile(
    r"(not (included|provided|contained|present|available|mentioned|covered|shown|enough)|"
    r"(does not|doesn't|do not|don't|cannot|can't|unable to|couldn't)\b.{0,50}\b(contain|include|provide|answer|determine|say|find|show|explain|describe)|"
    r"no (information|mention|details)|insufficient|outside (the|of) (the )?(provided|given|scope)|isn't (included|in))", re.I)


def load_rows():
    rows = [json.loads(l) for l in pathlib.Path(args.dataset).read_text().splitlines() if l.strip()]
    return rows[: args.limit] if args.limit else rows


def jl_write(path, items):
    path.write_text("\n".join(json.dumps(i, default=str) for i in items) + "\n")


def jl_read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def mention_hits(answer: str, must) -> tuple[int, int]:
    a = answer.lower()
    got = sum(1 for m in must if any(alt.lower() in a for alt in ([m] if isinstance(m, str) else m)))
    return got, len(must)


# =========================================================================================
# Stage 1 — deterministic: file extraction + retrieval (no LLM)
# =========================================================================================
def stage_deterministic(rows, nodes):
    from app.chains.code_explanation_chain import (_get_call_graph, _get_collection, _retrieve_and_expand)
    from app.classification.classifier import extract_target_files

    coll, cg = _get_collection(), _get_call_graph()
    _retrieve_and_expand(coll, cg, "warm up the embedding model")  # exclude one-off model load from timings
    out = []
    for r in rows:
        rec = {"id": r["id"], "extracted_files": extract_target_files(r["query"])}
        gold = [nodes[g] for g in r["gold"]]
        if r["intent"] == "code_explanation":
            t = time.perf_counter()
            res = _retrieve_and_expand(coll, cg, r["query"])
            rec["retrieval_s"] = time.perf_counter() - t
            span = lambda d: (d.metadata["file_path"], d.metadata["start_line"], d.metadata["end_line"])
            prim, exp = [span(d) for d in res["primary_docs"]], [span(d) for d in res["expanded_docs"]]
            rec["primary_spans"], rec["expanded_spans"] = prim, exp
            # naive baseline: plain top-3 chunks by embedding similarity, no dedupe/boost/expansion
            raw = coll.query(query_texts=[r["query"]], n_results=3)["metadatas"][0]
            naive = [(m["file_path"], m["start_line"], m["end_line"]) for m in raw if m["name"] != "<module>"]
            rec["naive_spans"] = naive
            if gold:
                overlap = lambda s, g: s[0] == g[0] and s[1] <= g[2] and s[2] >= g[1]
                rec["hit_primary"] = any(overlap(s, g) for s in prim for g in gold)
                rec["hit_naive"] = any(overlap(s, g) for s in naive for g in gold)
                rec["recall_context"] = sum(any(covers(*s, g) for s in prim + exp) for g in gold) / len(gold)
                rec["recall_naive"] = sum(any(covers(*s, g) for s in naive) for g in gold) / len(gold)
                rec["gold_in_primary_complete"] = all(any(covers(*s, g) for s in prim) for g in gold)
        out.append(rec)
        print(f"  [det] {r['id']} {r['category']:<12} files={rec['extracted_files']}", flush=True)
    return out


# =========================================================================================
# Stage 2 — pipeline: the real graph, end to end
# =========================================================================================
def stage_pipeline(rows):
    from langchain_core.callbacks import UsageMetadataCallbackHandler
    from app.errors import AppError
    from app.orchestration.graph import build_graph

    graph = build_graph()
    cold = None
    t = time.perf_counter()
    graph.invoke({"question": "What does the Auth class do?"})  # warm-up: loads model, index, call graph, git risk cache
    cold = time.perf_counter() - t
    print(f"  [pipe] cold start (first request, incl. model/index load): {cold:.1f}s", flush=True)

    out = []
    for r in rows:
        rec = {"id": r["id"], "attempts": 0, "error": None}
        for attempt in range(1, args.max_retries + 2):
            rec["attempts"] = attempt
            cb = UsageMetadataCallbackHandler()
            events, state = [], {}
            try:
                t0 = time.perf_counter()
                for upd in graph.stream({"question": r["query"]}, config={"callbacks": [cb]}, stream_mode="updates"):
                    now = time.perf_counter()
                    for node, delta in upd.items():
                        events.append((node, now - t0))
                        state.update(delta or {})
                rec["latency_s"] = time.perf_counter() - t0
                rec["error"] = None
                break
            except Exception as exc:  # AppError or anything unexpected: record, maybe retry
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                rec["error_retryable"] = bool(getattr(exc, "retryable", False))
                if attempt <= args.max_retries and isinstance(exc, AppError) and exc.retryable:
                    print(f"  [pipe] {r['id']} {rec['error'][:80]} -> wait {args.retry_wait}s, retry", flush=True)
                    time.sleep(args.retry_wait)
                    continue
                break
        if rec["error"] is None:
            fo = state["final_output"]
            rec.update(intent=fo.intent, answer=fo.answer, citations=list(fo.citations),
                       risk_details=None if fo.risk_details is None else [d.model_dump() for d in fo.risk_details])
            marks = dict(events)
            rec["stage_s"] = {"classify": marks["classify"],
                              "branch": max(v for k, v in marks.items() if k.endswith("_node")) - marks["classify"],
                              "format": marks["format_response"] - max(v for k, v in marks.items() if k.endswith("_node"))}
            usage = cb.usage_metadata
            rec["models_used"] = sorted(usage)
            rec["tokens"] = {"input": sum(u.get("input_tokens", 0) for u in usage.values()),
                             "output": sum(u.get("output_tokens", 0) for u in usage.values()),
                             "by_model": usage}
            rec["llm_calls_models"] = list(usage)
        out.append(rec)
        status = rec["error"] or f"{rec['intent']} {rec['latency_s']:.1f}s tok={rec['tokens']['input']}+{rec['tokens']['output']}"
        print(f"  [pipe] {r['id']} {r['category']:<12} {status}", flush=True)
        time.sleep(args.sleep)
    return out, cold


# =========================================================================================
# Stage 3 — judge: claim-level faithfulness against the exact context the generator saw
# =========================================================================================
JUDGE_PROMPT = """You are auditing an answer about a Python codebase for FAITHFULNESS to the source-code CONTEXT it was given.

1. Break the ANSWER into atomic factual claims about what the code does or contains. Skip hedges, greetings, and trailing
   "Note: ... risk" lines.
2. For each claim give a verdict:
   - supported:     stated in, or directly entailed by, the CONTEXT code
   - unsupported:   not established by the CONTEXT, even if it is plausibly true of the library in general
   - contradicted:  the CONTEXT says otherwise
   Judge strictly against the CONTEXT, not your own knowledge of the library.
3. abstains: true if the ANSWER explicitly says the provided context does not contain enough information to answer
   the question (a partial answer that also states a limitation is still false unless it declines the core question).

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}"""


def stage_judge(rows, pipe, judge_model):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel
    from typing import Literal
    from app.chains.code_explanation_chain import (_format_context, _get_call_graph, _get_collection, _retrieve_and_expand)
    from app.llm_utils import FALLBACK_MODEL_NAME, FallbackRunnable

    class Claim(BaseModel):
        claim: str
        verdict: Literal["supported", "unsupported", "contradicted"]
        evidence: str = ""

    class Judgement(BaseModel):
        claims: list[Claim]
        abstains: bool

    prompt = ChatPromptTemplate.from_messages([("human", JUDGE_PROMPT)])
    judge = FallbackRunnable(prompt | ChatGoogleGenerativeAI(model=judge_model).with_structured_output(Judgement),
                             prompt | ChatGoogleGenerativeAI(model=FALLBACK_MODEL_NAME).with_structured_output(Judgement),
                             FALLBACK_MODEL_NAME)
    coll, cg = _get_collection(), _get_call_graph()
    by_id = {p["id"]: p for p in pipe}
    out = []
    for r in rows:
        p = by_id[r["id"]]
        if p.get("error") or p.get("intent") != "code_explanation":
            continue
        ctx = _format_context(_retrieve_and_expand(coll, cg, r["query"]))["context"]  # deterministic replay of what the LLM saw
        rec = {"id": r["id"], "context": ctx}
        for attempt in range(1, args.max_retries + 2):
            try:
                j = judge.invoke({"question": r["query"], "context": ctx, "answer": p["answer"]})
                rec["judge"] = j.model_dump()
                break
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                if attempt <= args.max_retries:
                    time.sleep(args.retry_wait)
        out.append(rec)
        j = rec.get("judge")
        print(f"  [judge] {r['id']} " + (f"claims={len(j['claims'])} flagged={sum(c['verdict']!='supported' for c in j['claims'])} abstains={j['abstains']}" if j else rec.get("error", "?")), flush=True)
        time.sleep(args.sleep)
    return out


# =========================================================================================
# Stage 4 — metrics + report
# =========================================================================================
def parse_citations(cites, nodes):
    parsed = []
    for c in cites:
        m = CITATION_RE.match(c)
        if not m:
            parsed.append({"raw": c, "parsed": False})
            continue
        key = f"{m['file']}::{m['loc']}"
        node = nodes.get(key)
        a, b = int(m["a"]), int(m["b"])
        parsed.append({"raw": c, "parsed": True, "file": m["file"], "start": a, "end": b, "resolves": node is not None,
                       "line_exact": bool(node) and (node[1], node[2]) == (a, b),
                       "in_bounds": bool(node) and node[1] <= a <= b <= node[2]})
    return parsed


def compute(rows, nodes, det, pipe, judged, cold, vocab, risk_truth):
    D, P, J = ({x["id"]: x for x in xs} for xs in (det, pipe, judged))
    S = {"n_queries": len(rows), "by_category": dict(Counter(r["category"] for r in rows))}
    fails = []  # (id, kind, detail)

    # --- reliability
    ok = [r for r in rows if not P[r["id"]].get("error")]
    S["reliability"] = {"completed": len(ok), "errors": len(rows) - len(ok),
                        "retried_queries": sum(P[r["id"]]["attempts"] > 1 for r in rows),
                        "fallback_model_used": sum(len(P[r["id"]].get("models_used", [])) > 2 or any("3.1" in m for m in P[r["id"]].get("models_used", [])) for r in ok),
                        "error_list": [(r["id"], P[r["id"]]["error"]) for r in rows if P[r["id"]].get("error")]}

    # --- intent
    ik = sum(P[r["id"]]["intent"] == r["intent"] for r in ok)
    S["intent"] = {"correct": ik, "n": len(ok)}
    S["intent_by_category"] = {}
    for cat in S["by_category"]:
        rs = [r for r in ok if r["category"] == cat]
        S["intent_by_category"][cat] = [sum(P[r["id"]]["intent"] == r["intent"] for r in rs), len(rs)]
    for r in ok:
        if P[r["id"]]["intent"] != r["intent"]:
            fails.append((r["id"], "intent", f'"{r["query"]}" gold={r["intent"]} got={P[r["id"]]["intent"]}'))

    # --- target-file extraction (deterministic), on risk-intent queries
    risk_rows = [r for r in rows if r["intent"] == "risk_assessment"]
    ek = sum(D[r["id"]]["extracted_files"] == r["target_files"] for r in risk_rows)
    S["file_extraction"] = {"exact_match": ek, "n": len(risk_rows)}
    for r in risk_rows:
        if D[r["id"]]["extracted_files"] != r["target_files"]:
            fails.append((r["id"], "file_extraction", f'"{r["query"]}" gold={r["target_files"]} got={D[r["id"]]["extracted_files"]}'))

    # --- risk answers: routed, right files, numbers match independent recomputation, clarification behaves
    rk = [r for r in risk_rows if r["expect"] == "answer" and not P[r["id"]].get("error")]
    right_files = right_numbers = 0
    for r in rk:
        details = P[r["id"]].get("risk_details") or []
        files = [d["file"] for d in details]
        good_files = P[r["id"]]["intent"] == "risk_assessment" and files == r["target_files"]
        right_files += good_files
        good_nums = good_files and all(
            d["dependents"] == risk_truth[d["file"]]["dependents"] and d["commit_count"] == risk_truth[d["file"]]["commit_count"]
            and d["risk_level"] == risk_truth[d["file"]]["risk_level"] for d in details)
        right_numbers += good_nums
        if not good_nums:
            fails.append((r["id"], "risk", f'"{r["query"]}" files={files} gold={r["target_files"]}'))
    S["risk"] = {"right_files": right_files, "right_numbers": right_numbers, "n": len(rk)}
    cl = [r for r in rows if r["expect"] == "clarify" and not P[r["id"]].get("error")]
    clk = sum(P[r["id"]]["intent"] == "risk_assessment" and P[r["id"]].get("risk_details") is None and "which file" in P[r["id"]]["answer"].lower() for r in cl)
    S["clarification"] = {"correct": clk, "n": len(cl)}
    for r in cl:
        p = P[r["id"]]
        if not (p["intent"] == "risk_assessment" and p.get("risk_details") is None and "which file" in p["answer"].lower()):
            fails.append((r["id"], "clarification", f'"{r["query"]}" intent={p["intent"]} answer={p["answer"][:100]!r}'))
    S["risk_table"] = risk_truth

    # --- retrieval (deterministic stage)
    gold_rows = [r for r in rows if r["gold"] and r["intent"] == "code_explanation"]
    def rate(key, rs): return sum(bool(D[r["id"]][key]) for r in rs)
    S["retrieval"] = {"n": len(gold_rows),
                      "hit_system": rate("hit_primary", gold_rows), "hit_naive": rate("hit_naive", gold_rows),
                      "recall_system": statistics.mean(D[r["id"]]["recall_context"] for r in gold_rows) if gold_rows else None,
                      "recall_naive": statistics.mean(D[r["id"]]["recall_naive"] for r in gold_rows) if gold_rows else None,
                      "by_category": {}}
    for cat in ("named", "conceptual", "ambiguous"):
        rs = [r for r in gold_rows if r["category"] == cat]
        if rs:
            S["retrieval"]["by_category"][cat] = {"n": len(rs), "hit_system": rate("hit_primary", rs), "hit_naive": rate("hit_naive", rs)}
    for r in gold_rows:
        if not D[r["id"]]["hit_primary"]:
            fails.append((r["id"], "retrieval", f'"{r["query"]}" gold={r["gold"]} primary_spans={D[r["id"]]["primary_spans"]}'))
    S["latency_retrieval_s"] = {"p50": percentile([d["retrieval_s"] for d in det if "retrieval_s" in d], .5),
                                "p95": percentile([d["retrieval_s"] for d in det if "retrieval_s" in d], .95)}

    # --- answered code queries: citations, key facts, faithfulness
    ans_rows = [r for r in ok if r["expect"] == "answer" and r["intent"] == "code_explanation" and P[r["id"]]["intent"] == "code_explanation"]
    all_cites = []
    zero_cite = 0
    gold_cited = gold_cited_n = 0
    replay_match = replay_n = 0
    fact_full = fact_n = 0
    fact_frac = []
    for r in ans_rows:
        p = P[r["id"]]
        pc = parse_citations(p["citations"], nodes)
        all_cites += pc
        zero_cite += (len(pc) == 0)
        if r["gold"]:
            gold_cited_n += 1
            golds = [nodes[g] for g in r["gold"]]
            hit = any(c["parsed"] and any(covers(c["file"], c["start"], c["end"], g) for g in golds) for c in pc)
            gold_cited += hit
            if not hit:
                fails.append((r["id"], "citation_gold", f'"{r["query"]}" gold={r["gold"]} cited={p["citations"]}'))
        if r["id"] in D and "primary_spans" in D[r["id"]]:
            replay_n += 1
            replay = {tuple(s) for s in D[r["id"]]["primary_spans"] + D[r["id"]]["expanded_spans"]}
            actual = {(c["file"], c["start"], c["end"]) for c in pc if c["parsed"]}
            replay_match += (replay == actual)
        if r["must_mention"]:
            g, n = mention_hits(p["answer"], r["must_mention"])
            fact_n += 1
            fact_full += (g == n)
            fact_frac.append(g / n)
            if g < n:
                fails.append((r["id"], "key_facts", f'"{r["query"]}" mentioned {g}/{n} of {r["must_mention"]}'))
    S["citations"] = {"n_answers": len(ans_rows), "n_citations": len(all_cites),
                      "resolve_to_real_code": sum(c.get("resolves", False) for c in all_cites),
                      "line_exact": sum(c.get("line_exact", False) for c in all_cites),
                      "unparseable": sum(not c["parsed"] for c in all_cites),
                      "answers_with_zero_citations": zero_cite,
                      "gold_cited": gold_cited, "gold_cited_n": gold_cited_n}
    for r in ans_rows:
        for c in parse_citations(P[r["id"]]["citations"], nodes):
            if not c.get("line_exact", False):
                fails.append((r["id"], "citation_validity", f'{c["raw"]} resolves={c.get("resolves")} in_bounds={c.get("in_bounds")}'))
    S["context_replay"] = {"match": replay_match, "n": replay_n}
    S["key_facts"] = {"all_present": fact_full, "n": fact_n, "mean_fraction": statistics.mean(fact_frac) if fact_frac else None}

    # --- deterministic identifier grounding (applies to every LLM-written answer, incl. out-of-scope)
    llm_rows = [r for r in ok if P[r["id"]]["intent"] == "code_explanation"]
    n_out_repo = n_out_ctx = 0
    ident_detail = {}
    for r in llm_rows:
        ctx = J.get(r["id"], {}).get("context")
        if ctx is None:
            continue
        ans = P[r["id"]]["answer"].split("\n\nNote: ")[0]
        cands = set()
        for span in re.findall(r"`([^`\n]+)`", ans):
            cands |= {t for t in tokens(span) if len(t) >= 3}
        cands |= set(re.findall(r"\b[A-Za-z]+_[A-Za-z0-9_]+\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b", ans))
        base = vocab_base = set()
        not_repo = sorted(c for c in cands if c not in vocab)
        not_ctx = sorted(c for c in cands if c not in tokens(ctx) and c in vocab)
        ident_detail[r["id"]] = {"not_in_repo": not_repo, "not_in_context": not_ctx}
        n_out_repo += bool(not_repo)
        n_out_ctx += bool(not_ctx)
    S["identifiers"] = {"n": len(ident_detail), "answers_with_identifier_not_in_httpx": n_out_repo,
                        "answers_with_identifier_outside_context": n_out_ctx, "detail": ident_detail}

    # --- judge: claim-level faithfulness + abstention
    jr = [r for r in ans_rows if J.get(r["id"], {}).get("judge")]
    claims = [c for r in jr for c in J[r["id"]]["judge"]["claims"]]
    flagged = [c for c in claims if c["verdict"] != "supported"]
    S["faithfulness"] = {"n_answers": len(jr), "n_claims": len(claims), "unsupported": sum(c["verdict"] == "unsupported" for c in claims),
                         "contradicted": sum(c["verdict"] == "contradicted" for c in claims),
                         "answers_with_flagged_claim": sum(any(c["verdict"] != "supported" for c in J[r["id"]]["judge"]["claims"]) for r in jr),
                         "by_category": {}}
    for cat in ("named", "conceptual", "ambiguous"):
        rs = [r for r in jr if r["category"] == cat]
        if rs:
            S["faithfulness"]["by_category"][cat] = [sum(any(c["verdict"] != "supported" for c in J[r["id"]]["judge"]["claims"]) for r in rs), len(rs)]
    for r in jr:
        for c in J[r["id"]]["judge"]["claims"]:
            if c["verdict"] != "supported":
                fails.append((r["id"], "faithfulness/" + c["verdict"], c["claim"]))

    oos = [r for r in ok if r["expect"] == "abstain"]
    oos_j = [r for r in oos if J.get(r["id"], {}).get("judge")]
    S["abstention"] = {"n": len(oos),
                       "judge_abstains": sum(J[r["id"]]["judge"]["abstains"] for r in oos_j), "judged": len(oos_j),
                       "regex_abstains": sum(bool(ABSTAIN_RE.search(P[r["id"]]["answer"])) for r in oos),
                       "answered_with_unsupported_claims": sum(any(c["verdict"] != "supported" for c in J[r["id"]]["judge"]["claims"]) for r in oos_j if not J[r["id"]]["judge"]["abstains"]),
                       "with_citations_attached": sum(len(P[r["id"]]["citations"]) > 0 for r in oos)}
    for r in oos_j:
        if not J[r["id"]]["judge"]["abstains"]:
            fails.append((r["id"], "abstention", f'"{r["query"]}" did not abstain; answer={P[r["id"]]["answer"][:160]!r}'))

    # --- latency / tokens
    lat = lambda rs: [P[r["id"]]["latency_s"] for r in rs]
    S["latency_s"] = {"cold_start": cold, "overall": {"n": len(ok), "p50": percentile(lat(ok), .5), "p95": percentile(lat(ok), .95), "mean": statistics.mean(lat(ok)) if ok else None}}
    for name, rs in (("code_explanation", [r for r in ok if P[r["id"]]["intent"] == "code_explanation"]),
                     ("risk_assessment", [r for r in ok if P[r["id"]]["intent"] == "risk_assessment"])):
        if rs:
            S["latency_s"][name] = {"n": len(rs), "p50": percentile(lat(rs), .5), "p95": percentile(lat(rs), .95), "mean": statistics.mean(lat(rs))}
            S["latency_s"][name]["stages_mean"] = {k: statistics.mean(P[r["id"]]["stage_s"][k] for r in rs) for k in ("classify", "branch", "format")}
    tok = lambda rs, k: statistics.mean(P[r["id"]]["tokens"][k] for r in rs) if rs else 0
    S["tokens_per_query"] = {"all": {"input": tok(ok, "input"), "output": tok(ok, "output")}}
    for name in ("code_explanation", "risk_assessment"):
        rs = [r for r in ok if P[r["id"]]["intent"] == name]
        S["tokens_per_query"][name] = {"n": len(rs), "input": tok(rs, "input"), "output": tok(rs, "output")}
    if args.price_in is not None and args.price_out is not None:
        S["cost_usd_per_query"] = {k: (v["input"] * args.price_in + v["output"] * args.price_out) / 1e6 for k, v in S["tokens_per_query"].items()}
    return S, fails


def write_report(S, fails, rows, outdir, meta):
    L = []
    w = L.append
    I, R = S["intent"], S["retrieval"]
    w(f"# RepoSage evaluation: {S['n_queries']} labelled queries\n")
    w(f"Run {meta['when']} · generator `{meta['generator']}` (fallback `{meta['fallback']}`) · judge `{meta['judge']}` · "
      f"httpx @ `{meta['httpx_commit'][:10]}` · tracing off\n")
    w("Rates are shown as k/n with 95% Wilson intervals: with n≈48 a single query moves a rate by ~2 points, so read the interval, not the point estimate.\n")
    w("## Headline\n")
    w("| Metric | Result |\n|---|---|")
    w(f"| Intent routing accuracy | {pct(I['correct'], I['n'])} |")
    fe = S["file_extraction"]
    w(f"| Target-file extraction, exact match (deterministic, risk queries) | {pct(fe['exact_match'], fe['n'])} |")
    rk = S["risk"]
    w(f"| Risk answers with the right files | {pct(rk['right_files'], rk['n'])} |")
    w(f"| Risk answers whose numbers + level match an independent recomputation | {pct(rk['right_numbers'], rk['n'])} |")
    cl = S["clarification"]
    w(f"| Asks for a file instead of guessing (no file named) | {pct(cl['correct'], cl['n'])} |")
    w(f"| Retrieval: gold function in the context (RepoSage) | {pct(R['hit_system'], R['n'])} |")
    w(f"| Retrieval: same, naive top-3 chunk baseline | {pct(R['hit_naive'], R['n'])} |")
    c = S["citations"]
    w(f"| Citations that point at real code (file+function exist) | {pct(c['resolve_to_real_code'], c['n_citations'])} |")
    w(f"| Citations with exact line range | {pct(c['line_exact'], c['n_citations'])} |")
    w(f"| Answers that cite the gold function | {pct(c['gold_cited'], c['gold_cited_n'])} |")
    f_ = S["faithfulness"]
    if f_["n_answers"]:
        w(f"| Hallucination rate, response level (≥1 unsupported/contradicted claim) | {pct(f_['answers_with_flagged_claim'], f_['n_answers'])} |")
        w(f"| Hallucination rate, claim level | {pct(f_['unsupported'] + f_['contradicted'], f_['n_claims'])} |")
    a = S["abstention"]
    if a["judged"]:
        w(f"| Abstains on out-of-scope questions | {pct(a['judge_abstains'], a['judged'])} |")
    lt = S["latency_s"]
    w(f"| End-to-end latency p50 / p95 | {lt['overall']['p50']:.1f}s / {lt['overall']['p95']:.1f}s (n={lt['overall']['n']}, warm) |")
    t = S["tokens_per_query"]["all"]
    cost = S.get("cost_usd_per_query")
    w(f"| Tokens per query (mean) | {t['input']:.0f} in + {t['output']:.0f} out" + (f" ≈ ${cost['all']:.5f}/query" if cost else " (no price supplied: cost not computed)") + " |")
    w(f"| Completed without error | {pct(S['reliability']['completed'], S['n_queries'])} |\n")

    w("## Intent routing by category\n")
    w("| Category | Correct |\n|---|---|")
    for cat, (k, n) in S["intent_by_category"].items():
        w(f"| {cat} | {pct(k, n)} |")
    w("\n## Retrieval vs. a naive baseline\n")
    w("`hit` = a retrieved span overlaps the gold function. Baseline = plain top-3 chunks by embedding similarity (no overfetch/dedupe, exact-identifier boost, split reassembly, or call-graph expansion). Overlap is generous to the baseline: a fragment of a split function counts.\n")
    w("| Slice | n | RepoSage hit | Baseline hit |\n|---|---|---|---|")
    w(f"| all | {R['n']} | {pct(R['hit_system'], R['n'])} | {pct(R['hit_naive'], R['n'])} |")
    for cat, v in R["by_category"].items():
        w(f"| {cat} | {v['n']} | {pct(v['hit_system'], v['n'])} | {pct(v['hit_naive'], v['n'])} |")
    if R["recall_system"] is not None:
        w(f"\nMean gold recall in the full context (primary + 1-hop expansion): RepoSage {R['recall_system']:.2f} vs baseline {R['recall_naive']:.2f}.\n")

    w("## Latency (warm) and cost\n")
    w("| Path | n | p50 | p95 | classify | branch | format |\n|---|---|---|---|---|---|---|")
    for name in ("code_explanation", "risk_assessment"):
        if name in lt:
            v = lt[name]; st = v["stages_mean"]
            w(f"| {name} | {v['n']} | {v['p50']:.1f}s | {v['p95']:.1f}s | {st['classify']:.1f}s | {st['branch']:.1f}s | {st['format']:.2f}s |")
    w(f"\nCold start (first request: embedding model + index + call graph + git risk cache): {lt['cold_start']:.1f}s. Retrieval alone (warm): p50 {S['latency_retrieval_s']['p50']*1000:.0f} ms, p95 {S['latency_retrieval_s']['p95']*1000:.0f} ms.\n")
    w("| Path | n | input tok | output tok |\n|---|---|---|---|")
    for name in ("code_explanation", "risk_assessment"):
        v = S["tokens_per_query"][name]
        w(f"| {name} | {v['n']} | {v['input']:.0f} | {v['output']:.0f} |")
    w("")

    w("## Faithfulness detail\n")
    if f_["n_answers"]:
        w(f"- Judged {f_['n_answers']} generated answers, {f_['n_claims']} atomic claims: {f_['unsupported']} unsupported, {f_['contradicted']} contradicted.")
        for cat, (k, n) in f_["by_category"].items():
            w(f"- {cat}: {k}/{n} answers had a flagged claim")
    idn = S["identifiers"]
    w(f"- Deterministic check (no LLM): {idn['answers_with_identifier_not_in_httpx']}/{idn['n']} answers name an identifier that appears nowhere in the httpx package; {idn['answers_with_identifier_outside_context']}/{idn['n']} name one that is in httpx but absent from the retrieved context.")
    if a["n"]:
        w(f"- Out-of-scope queries ({a['n']}): judge says abstained {a['judge_abstains']}/{a['judged']}; phrase-matcher says {a['regex_abstains']}/{a['n']}; "
          f"{a['with_citations_attached']}/{a['n']} responses still attached citations to retrieved-but-irrelevant code.")
    w(f"- Context replay check: re-running retrieval reproduces the exact cited spans in {S['context_replay']['match']}/{S['context_replay']['n']} runs (validates that the judge saw what the generator saw).")
    kf = S["key_facts"]
    if kf["n"]:
        w(f"- Key-fact coverage: all expected facts present in {pct(kf['all_present'], kf['n'])}; mean fraction {kf['mean_fraction']:.2f}.")
    w("\n## Independent risk recomputation (ast imports + `git rev-list --count`)\n")
    w("| file | dependents | commits | level |\n|---|---|---|---|")
    for f, v in sorted(S["risk_table"].items(), key=lambda x: -x[1]["dependents"]):
        w(f"| {f} | {v['dependents']} | {v['commit_count']} | {v['risk_level']} |")

    w("\n## Failures and flags (all of them, not a sample)\n")
    if not fails:
        w("None.")
    kinds = defaultdict(list)
    for qid, kind, detail in fails:
        kinds[kind].append((qid, detail))
    for kind, items in sorted(kinds.items()):
        w(f"\n**{kind}** ({len(items)})\n")
        for qid, detail in items:
            w(f"- `{qid}` {detail}")
    w("\n## Method and limits\n")
    w("""- Labels: authored from the httpx source before any system output was seen (`eval/build_dataset.py`), then machine-checked against the source (`eval/validate_dataset.py`).
- Hit/recall use line-range overlap/containment against AST ranges, so a class chunk correctly covers its methods and a split function is judged on its reassembled span.
- Faithfulness uses an LLM judge from the same model family as the generator, which can share its blind spots. It is a screen, not ground truth: the verdicts on every flagged claim were hand-checked (see `eval/results/latest/judge_review.md`).
- Risk numbers are checked against a separate implementation of the documented rule (transitive importers, commit counts, median gate). That validates the implementation against the spec, not the spec's choice of rule.
- 48 queries on one repo: good for catching regressions and comparing variants, not for population-level claims.""")
    (outdir / "REPORT.md").write_text("\n".join(L) + "\n")


def main():
    rows = load_rows()
    nodes = load_nodes()
    outdir = pathlib.Path(args.resume or args.out or HERE / "results" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"results -> {outdir}")

    from app.chains.code_explanation_chain import LLM_MODEL_NAME
    from app.llm_utils import FALLBACK_MODEL_NAME
    judge_model = args.judge_model or LLM_MODEL_NAME

    f_det, f_pipe, f_judge, f_meta = (outdir / n for n in ("deterministic.jsonl", "pipeline.jsonl", "judge.jsonl", "meta.json"))

    if f_det.exists(): det = jl_read(f_det)
    else:
        print("stage 1: deterministic (file extraction + retrieval)")
        det = stage_deterministic(rows, nodes); jl_write(f_det, det)

    if f_pipe.exists():
        pipe = jl_read(f_pipe); cold = json.loads(f_meta.read_text())["cold_start_s"]
    else:
        print("stage 2: pipeline (real graph, real Gemini)")
        pipe, cold = stage_pipeline(rows); jl_write(f_pipe, pipe)
        f_meta.write_text(json.dumps({"cold_start_s": cold}))

    if f_judge.exists(): judged = jl_read(f_judge)
    elif args.skip_judge: judged = []
    else:
        print("stage 3: faithfulness judge")
        judged = stage_judge(rows, pipe, judge_model); jl_write(f_judge, judged)

    print("stage 4: metrics + report")
    import subprocess
    commit = subprocess.run(["git", "-C", str(_R / "target_repos/httpx"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    S, fails = compute(rows, nodes, det, pipe, judged, cold, repo_vocabulary(), independent_risk())
    meta = {"when": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "generator": LLM_MODEL_NAME, "fallback": FALLBACK_MODEL_NAME,
            "judge": judge_model if judged else "skipped", "httpx_commit": commit}
    (outdir / "summary.json").write_text(json.dumps({"meta": meta, "summary": S, "failures": fails}, indent=2, default=str))
    write_report(S, fails, rows, outdir, meta)
    print(f"\nwrote {outdir/'REPORT.md'}")


if __name__ == "__main__":
    main()
