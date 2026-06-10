#!/usr/bin/env python3
"""
Extract per-role baseline system prompts + a roles.json manifest from the v2
training corpus.

For each Archon role (meta.role) this:
  - groups every training record's system prompt,
  - derives a baseline TEMPLATE by finding the invariant span (longest common
    prefix + suffix across the role's prompts) and marking the scenario-variable
    middle with {{SCENARIO_CONTEXT}},
  - auto-detects output_type (json vs markdown) by parsing the gold assistant
    outputs, and for json roles infers the top-level schema + enum values,
  - writes prompts/<role>/baseline_template.txt and a verbatim sample,
  - emits roles.json (the manifest the runner + scorer read).

It also prints a diagnostic so we can SEE how cleanly each role templates
(invariant vs variable span) rather than assuming. No GPU, no network.

Usage: python promptlab/extract_baselines.py [--train data/build/archon_v2.train.jsonl]
"""
from __future__ import annotations
import argparse, json, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROMPTS = HERE / "prompts"

# A role is "cleanly templatable" if its invariant (non-scenario) span is at
# least this fraction of the shortest prompt — i.e. the scenario-variable middle
# didn't swallow the role instructions.
TEMPLATABLE_MIN_INVARIANT_FRAC = 0.4
SCENARIO_SLOT = "{{SCENARIO_CONTEXT}}"


def longest_common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    s1, s2 = min(strings), max(strings)
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


def longest_common_suffix(strings: list[str]) -> str:
    rev = [s[::-1] for s in strings]
    return longest_common_prefix(rev)[::-1]


def detect_output_type(outputs: list[str]) -> tuple[str, dict]:
    """Return ('json'|'markdown', schema_info). json iff a majority parse."""
    parsed = []
    for o in outputs:
        try:
            parsed.append(json.loads(o.strip()))
        except (json.JSONDecodeError, ValueError):
            parsed.append(None)
    n_json = sum(p is not None and isinstance(p, dict) for p in parsed)
    if n_json < 0.5 * len(outputs):
        return "markdown", {}
    # Infer schema: union of keys; enum values for short-string fields.
    keys = collections.Counter()
    field_values: dict[str, set] = collections.defaultdict(set)
    for p in parsed:
        if not isinstance(p, dict):
            continue
        for k, v in p.items():
            keys[k] += 1
            if isinstance(v, str) and len(v) <= 24:
                field_values[k].add(v)
            elif v is None:
                field_values[k].add(None)
    schema = {"required_keys": sorted(keys), "enums": {}}
    for k, vals in field_values.items():
        # treat as enum if few distinct short values
        if 0 < len(vals) <= 8:
            schema["enums"][k] = sorted(v for v in vals if v is not None)
    return "json", schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(ROOT / "data/build/archon_v2.train.jsonl"))
    args = ap.parse_args()

    by_role_sys: dict[str, list[str]] = collections.defaultdict(list)
    by_role_out: dict[str, list[str]] = collections.defaultdict(list)
    for line in open(args.train):
        if not line.strip():
            continue
        r = json.loads(line)
        role = r["meta"].get("role")
        if role is None:
            continue
        c = r["conversations"]
        by_role_sys[role].append(c[0]["content"])
        by_role_out[role].append(c[-1]["content"])

    manifest = {}
    print(f"{'role':24} {'type':9} {'n':>4} {'inv%':>5} {'var_chars':>9}  templatable")
    print("-" * 70)
    for role in sorted(by_role_sys):
        sys_prompts = by_role_sys[role]
        outputs = by_role_out[role]
        lcp = longest_common_prefix(sys_prompts)
        lcs = longest_common_suffix(sys_prompts)
        # guard: prefix+suffix must not overlap on the shortest prompt
        shortest = min(len(s) for s in sys_prompts)
        invariant = len(lcp) + len(lcs)
        if invariant > shortest:  # overlap -> all prompts identical-ish
            lcs = ""
            invariant = len(lcp)
        inv_frac = invariant / shortest if shortest else 0.0
        var_chars = shortest - invariant
        templatable = inv_frac >= TEMPLATABLE_MIN_INVARIANT_FRAC and len(lcp) > 0
        out_type, schema = detect_output_type(outputs)

        template = (lcp + SCENARIO_SLOT + lcs) if templatable else sys_prompts[0]
        role_dir = PROMPTS / role
        role_dir.mkdir(parents=True, exist_ok=True)
        (role_dir / "baseline_template.txt").write_text(template)
        (role_dir / "sample.txt").write_text(sys_prompts[0])

        manifest[role] = {
            "output_type": out_type,
            "n_train": len(sys_prompts),
            "templatable": templatable,
            "invariant_chars": invariant,
            "invariant_frac": round(inv_frac, 3),
            "variable_chars": var_chars,
            "scenario_slot": SCENARIO_SLOT if templatable else None,
            "schema": schema if out_type == "json" else None,
            "baseline_template": str((role_dir / "baseline_template.txt").relative_to(ROOT)),
        }
        print(f"{role:24} {out_type:9} {len(sys_prompts):>4} "
              f"{inv_frac*100:>4.0f}% {var_chars:>9}  {templatable}")

    (HERE / "roles.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {HERE / 'roles.json'} ({len(manifest)} roles)")
    # surface json schemas for a quick eyeball
    print("\n=== JSON role schemas ===")
    for role, m in manifest.items():
        if m["output_type"] == "json":
            print(f"  {role}: keys={m['schema']['required_keys']} enums={m['schema']['enums']}")


if __name__ == "__main__":
    main()
