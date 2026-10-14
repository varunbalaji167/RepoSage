"""Print every judge-flagged claim for hand audit. Usage: python eval/judge_review.py [results-dir]"""
import json, pathlib, re, sys, textwrap

HERE = pathlib.Path(__file__).resolve().parent
if len(sys.argv) > 1:
    D = pathlib.Path(sys.argv[1]).resolve()
else:
    D = sorted((HERE / "results").iterdir())[-1]

ds = {json.loads(l)["id"]: json.loads(l) for l in (HERE / "dataset.jsonl").read_text().splitlines()}
pipe = {json.loads(l)["id"]: json.loads(l) for l in (D / "pipeline.jsonl").read_text().splitlines()}
judge = {json.loads(l)["id"]: json.loads(l) for l in (D / "judge.jsonl").read_text().splitlines()}

flagged, total = 0, 0
for qid, j in sorted(judge.items()):
    if "judge" not in j: continue
    bad = [c for c in j["judge"]["claims"] if c["verdict"] != "supported"]
    total += len(j["judge"]["claims"])
    if not bad: continue
    flagged += len(bad)
    print(f"\n=== {qid} :: {ds[qid]['query']} ===")
    print(f"ANSWER: {textwrap.shorten(pipe[qid]['answer'], 400)}")
    for c in bad:
        print(f"\n  [{c['verdict']}] {c['claim']}")
        if c.get("evidence"): print(f"     evidence: {textwrap.shorten(c['evidence'], 240)}")
    files = sorted(set(re.findall(r"\[(_[a-z]+\.py)", j["context"])))
    print(f"\n  context from: {', '.join(files) or '(none parsed)'}")

print(f"\n---\n{flagged}/{total} claims flagged across {sum(1 for j in judge.values() if any(c['verdict']!='supported' for c in j.get('judge',{}).get('claims',[])))}/{len(judge)} answers.")
