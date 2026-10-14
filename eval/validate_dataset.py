"""Check every gold label against the real source: gold nodes exist, must_mention
terms are derivable from the gold code, target files are real, ids unique."""
import ast, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PKG = ROOT / "target_repos/httpx/httpx"
STEMS = {"_api","_client","_models","_config","_auth","_urls","_exceptions","_types","_utils"}

def node_sources():
    out = {}
    for stem in STEMS:
        src = (PKG / f"{stem}.py").read_text()
        for n in ast.parse(src).body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[f"{stem}.py::{n.name}"] = ast.get_source_segment(src, n)
            elif isinstance(n, ast.ClassDef):
                out[f"{stem}.py::{n.name}"] = ast.get_source_segment(src, n)
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out[f"{stem}.py::{n.name}.{m.name}"] = ast.get_source_segment(src, m)
    return out

def main():
    nodes = node_sources()
    rows = [json.loads(l) for l in (ROOT/"eval/dataset.jsonl").read_text().splitlines()]
    errs = []
    if len({r["id"] for r in rows}) != len(rows): errs.append("duplicate ids")
    for r in rows:
        for g in r["gold"]:
            if g not in nodes: errs.append(f'{r["id"]}: gold node not found: {g}')
        if r["gold"]:
            text = "\n".join(nodes.get(g, "") for g in r["gold"]).lower()
            for t in r["must_mention"]:
                alts = [t] if isinstance(t, str) else t
                if not any(a.lower() in text for a in alts):
                    errs.append(f'{r["id"]}: must_mention {t!r} not derivable from gold source')
        for f in r["target_files"]:
            if f not in STEMS: errs.append(f'{r["id"]}: unknown target file {f}')
        if r["intent"] not in ("code_explanation","risk_assessment"): errs.append(f'{r["id"]}: bad intent')
    print(f"{len(rows)} queries checked")
    for e in errs: print("ERROR", e)
    print("OK" if not errs else f"{len(errs)} error(s)")
    sys.exit(1 if errs else 0)
main()
