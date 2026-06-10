#!/usr/bin/env python3
"""
PromptLab report — aggregate per-(role,variant) scores into a leaderboard.

Input: a scores jsonl, one row per (variant × record):
  {"variant","role","idx","score", ...}  (rows are scorer.ScoreResult dicts
  plus variant/idx, as emitted by the scoring step).

Output: runs/<run>/report.json and report.md with, per role:
  - mean score per variant, sorted (winner first),
  - delta vs the "baseline" variant,
  - n and a low-confidence flag for thin roles (from eval/_index.json).

Pure python, no GPU, no network.

Usage: python promptlab/report.py --scores runs/promptlab_smoke/scores.jsonl --run smoke
"""
from __future__ import annotations
import argparse, json, collections, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def aggregate(rows: list[dict]) -> dict:
    # role -> variant -> [scores]
    acc: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        acc[r["role"]][r["variant"]].append(float(r["score"]))
    out = {}
    for role, variants in acc.items():
        means = {v: round(statistics.mean(s), 4) for v, s in variants.items()}
        base = means.get("baseline")
        board = []
        for v, m in sorted(means.items(), key=lambda x: -x[1]):
            board.append({"variant": v, "mean": m, "n": len(variants[v]),
                          "delta_vs_baseline": (round(m - base, 4) if base is not None else None)})
        out[role] = {"baseline": base, "winner": board[0]["variant"], "leaderboard": board}
    return out


def render_md(agg: dict, index: dict) -> str:
    lines = ["# PromptLab report", ""]
    low = {r: v.get("low_confidence") for r, v in index.get("roles", {}).items()}
    for role in sorted(agg):
        flag = "  ⚠️ low-confidence (thin eval)" if low.get(role) else ""
        a = agg[role]
        lines.append(f"## {role}{flag}")
        lines.append("")
        lines.append("| rank | variant | mean | Δ vs baseline | n |")
        lines.append("|---|---|---|---|---|")
        for i, row in enumerate(a["leaderboard"], 1):
            d = row["delta_vs_baseline"]
            dstr = "—" if d is None else (f"+{d}" if d >= 0 else f"{d}")
            mark = " 🏆" if row["variant"] == a["winner"] and len(a["leaderboard"]) > 1 else ""
            lines.append(f"| {i} | `{row['variant']}`{mark} | {row['mean']} | {dstr} | {row['n']} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--run", default="smoke")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.scores).read_text().splitlines() if l.strip()]
    agg = aggregate(rows)
    index_f = HERE / "eval" / "_index.json"
    index = json.loads(index_f.read_text()) if index_f.exists() else {"roles": {}}

    run_dir = HERE / "runs" / f"promptlab_{args.run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(agg, indent=2))
    md = render_md(agg, index)
    (run_dir / "report.md").write_text(md)
    print(md)
    print(f"\nwrote {run_dir/'report.json'} and report.md")


if __name__ == "__main__":
    main()
