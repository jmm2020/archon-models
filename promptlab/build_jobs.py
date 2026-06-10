#!/usr/bin/env python3
"""
PromptLab job builder — turn factored scenarios + candidate templates into a
runner jobs file.

Inputs (per role):
  factored/<role>.json   {"template": "...{{SCENARIO_CONTEXT}}...",
                          "records": [{"idx": 0, "scenario_context": "..."}, ...]}
                         (written by the frontier factoring agent)
  candidates/<role>.json {"candidates": [{"name": "...", "template": "..."}, ...]}
                         (written by the frontier author agent)
  eval/<role>.jsonl      eval records (own verbatim system prompt = baseline)

Output: jobs.jsonl, one row per (variant × record):
  {"variant","role","idx","system","user"}

Variants = "baseline" (record's own system) + each candidate (template
instantiated with the record's scenario_context). A template with no
{{SCENARIO_CONTEXT}} slot is used verbatim (scenario-free roles, e.g. routing).

Also prints a validation report: slot counts, missing contexts, and the
similarity of (factored template + context) reconstruction vs each record's
real system prompt (diagnostic only — baseline always uses the verbatim prompt).
"""
from __future__ import annotations
import argparse, difflib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLOT = "{{SCENARIO_CONTEXT}}"


def instantiate(template: str, scenario_context: str) -> str:
    """Substitute the scenario slot; verbatim if the template has no slot."""
    if SLOT in template:
        return template.replace(SLOT, scenario_context or "")
    return template


def validate_role(role: str, factored: dict, candidates: list[dict],
                  records: list[dict]) -> list[str]:
    """Return a list of fatal problems (empty = ok)."""
    problems = []
    tmpl = factored.get("template", "")
    has_slot = SLOT in tmpl
    if has_slot and tmpl.count(SLOT) != 1:
        problems.append(f"{role}: factored template has {tmpl.count(SLOT)} slots (want 1)")
    ctx_by_idx = {r["idx"]: r.get("scenario_context", "") for r in factored.get("records", [])}
    for i in range(len(records)):
        if i not in ctx_by_idx:
            problems.append(f"{role}: record {i} missing from factored records")
        elif has_slot and not ctx_by_idx[i].strip():
            problems.append(f"{role}: record {i} has empty scenario_context")
    for c in candidates:
        n = c.get("template", "").count(SLOT)
        want = 1 if has_slot else 0
        if n != want:
            problems.append(f"{role}/{c.get('name')}: {n} slots (want {want})")
        if not c.get("name"):
            problems.append(f"{role}: candidate without a name")
    return problems


def reconstruction_ratio(factored: dict, records: list[dict]) -> float | None:
    """Mean similarity of template+context vs each record's real system prompt."""
    tmpl = factored.get("template", "")
    ctx_by_idx = {r["idx"]: r.get("scenario_context", "") for r in factored.get("records", [])}
    ratios = []
    for i, rec in enumerate(records):
        rebuilt = instantiate(tmpl, ctx_by_idx.get(i, ""))
        ratios.append(difflib.SequenceMatcher(None, rebuilt, rec["system"]).ratio())
    return round(sum(ratios) / len(ratios), 3) if ratios else None


def build_jobs(role: str, factored: dict, candidates: list[dict],
               records: list[dict]) -> list[dict]:
    jobs = []
    ctx_by_idx = {r["idx"]: r.get("scenario_context", "") for r in factored.get("records", [])}
    for i, rec in enumerate(records):
        jobs.append({"variant": "baseline", "role": role, "idx": i,
                     "system": rec["system"], "user": rec["user"]})
        for c in candidates:
            jobs.append({"variant": c["name"], "role": role, "idx": i,
                         "system": instantiate(c["template"], ctx_by_idx.get(i, "")),
                         "user": rec["user"]})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", nargs="*", default=["create-plan", "classify-issue", "routing"])
    ap.add_argument("--out", default=str(HERE / "jobs.jsonl"))
    args = ap.parse_args()

    all_jobs, all_problems = [], []
    for role in args.roles:
        factored = json.loads((HERE / "factored" / f"{role}.json").read_text())
        candidates = json.loads((HERE / "candidates" / f"{role}.json").read_text())["candidates"]
        records = [json.loads(l) for l in (HERE / "eval" / f"{role}.jsonl").read_text().splitlines() if l.strip()]
        problems = validate_role(role, factored, candidates, records)
        all_problems += problems
        if problems:
            for p in problems:
                print(f"[build_jobs] PROBLEM: {p}")
            continue
        ratio = reconstruction_ratio(factored, records)
        jobs = build_jobs(role, factored, candidates, records)
        all_jobs += jobs
        print(f"[build_jobs] {role}: {len(records)} records × (baseline + {len(candidates)}) "
              f"= {len(jobs)} jobs · reconstruction≈{ratio}")

    if all_problems:
        sys.exit(f"[build_jobs] {len(all_problems)} problems — jobs file NOT written")
    Path(args.out).write_text("".join(json.dumps(j) + "\n" for j in all_jobs))
    print(f"[build_jobs] wrote {len(all_jobs)} jobs -> {args.out}")


if __name__ == "__main__":
    main()
