#!/usr/bin/env python3
"""
Group the held-out eval corpus by Archon role into per-role eval files the
runner + scorer consume.

Each output record (eval/<role>.jsonl) carries everything needed to score a
candidate prompt on that input:
  - system   : the record's OWN verbatim system prompt (scenario-correct baseline)
  - user     : the user turn (the task input)
  - gold     : the gold assistant output (Opus teacher)
  - scenario : meta.scenario
  - role     : meta.role

Also writes eval/_index.json with per-role counts and a low-confidence flag for
thin roles. No GPU, no network.

Usage: python promptlab/build_eval_set.py [--eval data/build/archon_v2.eval.jsonl]
"""
from __future__ import annotations
import argparse, json, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL_DIR = HERE / "eval"

# Roles with fewer than this many eval records get a low-confidence flag.
THIN_ROLE_THRESHOLD = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(ROOT / "data/build/archon_v2.eval.jsonl"))
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    by_role: dict[str, list[dict]] = collections.defaultdict(list)
    skipped = 0
    for line in open(args.eval):
        if not line.strip():
            continue
        r = json.loads(line)
        role = r["meta"].get("role")
        if role is None:
            skipped += 1
            continue
        c = r["conversations"]
        roles_in = [t["role"] for t in c]
        if roles_in[0] != "system" or roles_in[-1] != "assistant":
            skipped += 1
            continue
        # user turn = first non-system, non-assistant; fall back to second turn
        user = next((t["content"] for t in c if t["role"] == "user"), c[1]["content"])
        by_role[role].append({
            "role": role,
            "scenario": r["meta"].get("scenario"),
            "system": c[0]["content"],
            "user": user,
            "gold": c[-1]["content"],
        })

    index = {"threshold": THIN_ROLE_THRESHOLD, "skipped": skipped, "roles": {}}
    print(f"{'role':24} {'n':>4}  flag")
    print("-" * 40)
    for role in sorted(by_role):
        recs = by_role[role]
        out = EVAL_DIR / f"{role}.jsonl"
        with open(out, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        thin = len(recs) < THIN_ROLE_THRESHOLD
        index["roles"][role] = {"n": len(recs), "low_confidence": thin,
                                "file": str(out.relative_to(ROOT))}
        print(f"{role:24} {len(recs):>4}  {'LOW-CONFIDENCE' if thin else ''}")

    (EVAL_DIR / "_index.json").write_text(json.dumps(index, indent=2))
    total = sum(v["n"] for v in index["roles"].values())
    print(f"\n{total} records across {len(index['roles'])} roles "
          f"({skipped} skipped) -> {EVAL_DIR}")


if __name__ == "__main__":
    main()
