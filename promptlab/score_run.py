#!/usr/bin/env python3
"""
PromptLab run scorer — score every output of a runner run against gold.

Reads runs/promptlab_<run>/outputs.jsonl, loads each output text, looks up the
matching eval record, and calls scorer.tiered_score. The Opus judge is wired as
a FILE-BACKED lookup: judgments.jsonl rows
  {"role","variant","idx","faithfulness","completeness","grounding","format"}
are produced out-of-band (Workflow judge agents) and injected per row. Markdown
rows with no judgment get deterministic-only scores and a flag note, matching
scorer's judge-optional contract.

Output: runs/promptlab_<run>/scores.jsonl — scorer.ScoreResult dicts + variant/idx.

Usage:
  python promptlab/score_run.py --run subset_v2 [--judgments runs/.../judgments.jsonl]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from scorer import tiered_score

HERE = Path(__file__).resolve().parent


def load_judgments(path: Path | None) -> dict:
    """(role, variant, idx) -> rubric dict."""
    if not path or not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        out[(j["role"], j["variant"], j["idx"])] = {
            k: j[k] for k in ("faithfulness", "completeness", "grounding", "format") if k in j}
    return out


def score_rows(manifest_rows: list[dict], run_dir: Path, eval_records: dict,
               manifest: dict, judgments: dict) -> list[dict]:
    rows = []
    for m in manifest_rows:
        role, variant, idx = m["role"], m["variant"], m["idx"]
        output = (run_dir / m["output_file"]).read_text()
        record = eval_records[role][idx]
        rubric = judgments.get((role, variant, idx))
        judge = (lambda _role, _out, _gold, r=rubric: r) if rubric else None
        res = tiered_score(record, output, manifest, judge=judge)
        if res.output_type == "markdown" and rubric is None:
            res.notes.append("no judgment available (deterministic-only)")
        row = res.as_dict()
        row.update({"variant": variant, "idx": idx})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judgments", help="jsonl of judge rubric rows (optional)")
    args = ap.parse_args()

    run_dir = HERE / "runs" / f"promptlab_{args.run}"
    manifest_rows = [json.loads(l) for l in (run_dir / "outputs.jsonl").read_text().splitlines() if l.strip()]
    manifest = json.loads((HERE / "roles.json").read_text())
    eval_records = {}
    for role in sorted({m["role"] for m in manifest_rows}):
        eval_records[role] = [json.loads(l) for l in (HERE / "eval" / f"{role}.jsonl").read_text().splitlines() if l.strip()]
    judgments = load_judgments(Path(args.judgments) if args.judgments else None)

    rows = score_rows(manifest_rows, run_dir, eval_records, manifest, judgments)
    out_f = run_dir / "scores.jsonl"
    out_f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_judged = sum(1 for r in rows if r.get("judge") is not None)
    print(f"[score_run] {len(rows)} rows scored ({n_judged} with judge) -> {out_f}")


if __name__ == "__main__":
    main()
