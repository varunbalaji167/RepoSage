# eval/

Harness for the 48-query labelled evaluation. See `docs/EVAL.md` for the full write-up (what it measures and what it does *not* prove). This README is just how to run it.

## Run

```bash
python eval/run_eval.py                          # full run, costs real Gemini quota
python eval/run_eval.py --limit 6 --skip-judge   # smoke test, no LLM-judging stage
python eval/run_eval.py --resume eval/results/<timestamp>  # rebuild report, keep stages
```

Results land in `eval/results/<timestamp>/`: `REPORT.md`, `summary.json`, and the 3 raw stage files (`deterministic.jsonl`, `pipeline.jsonl`, `judge.jsonl`).

## Files

| | |
|---|---|
| `dataset.jsonl` | the 48 labelled queries |
| `build_dataset.py` | author-time script — labels live here as code, not just data |
| `validate_dataset.py` | checks every label against the real httpx source. Must stay green. |
| `common.py` | gold-label helpers + independent risk recomputation (ast + git, zero imports from `app/`) |
| `run_eval.py` | the harness |
| `judge_review.py` | print every judge-flagged claim for hand audit |

## Options worth knowing

- `--max-retries 2` (default) — re-run a query that trips both models' rate limits.
- `--sleep 0.5` — pause between queries so free-tier quotas don't rate-limit back-to-back.
- `--trace` — leave LangSmith on (off by default; 48 runs would spam the project).
- `--price-in / --price-out` — $ per 1M tokens; if omitted, report shows tokens only.
