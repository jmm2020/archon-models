#!/usr/bin/env python3
"""
Tiered scorer for PromptLab.

A model output is scored against the gold (Opus teacher) output for the same
eval input. Two tiers:

  - Deterministic (this module, no network): role-type aware.
      * JSON roles  -> parse, required-key check, enum validity, field match
                       vs gold. Fully objective.
      * Markdown roles -> a GATE in [0,1]: required-section coverage (sections
                       inferred from the gold's headers), prompt-leakage penalty,
                       length sanity. Objective, cheap.
  - Judge (Opus, injected): a callable scoring markdown prose quality 1-5 on
      faithfulness / completeness / grounding / format. Optional here so the
      deterministic tier is unit-testable offline; the runner wires a real judge.

`tiered_score()` dispatches on output_type and returns a ScoreResult.
Pure functions, no global state.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional


# ---- JSON helpers ---------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_loose(text: str):
    """Parse JSON, tolerating ```json fences and surrounding whitespace.
    Returns the object or None."""
    if text is None:
        return None
    stripped = _FENCE.sub("", text.strip())
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        # last resort: first {...} span
        m = re.search(r"\{.*\}", stripped, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
        return None


# ---- result type ----------------------------------------------------------

@dataclass
class ScoreResult:
    role: str
    output_type: str
    score: float                       # final composite in [0,1]
    parses: Optional[bool] = None
    keys_ok: Optional[bool] = None
    enums_ok: Optional[bool] = None
    match: Optional[float] = None      # field/contract match vs gold [0,1]
    gate: Optional[float] = None       # markdown deterministic gate [0,1]
    judge: Optional[float] = None      # judge composite [0,1] if provided
    notes: list = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


# ---- JSON scoring ---------------------------------------------------------

# Roles whose correctness reduces to specific fields (not whole-object equality).
def _match_json(role: str, out: dict, gold: dict) -> tuple[float, list]:
    notes = []
    if role in ("routing", "escalation"):
        # workflow choice is the primary signal (0.7); escalate decision is a
        # separate, meaningful but secondary signal (0.3).
        wf_ok = out.get("workflow") == gold.get("workflow")
        esc_ok = bool(out.get("escalate")) == bool(gold.get("escalate"))
        score = 0.7 * wf_ok + 0.3 * esc_ok
        if not wf_ok:
            notes.append(f"workflow differs ({out.get('workflow')} vs {gold.get('workflow')})")
        if not esc_ok:
            notes.append("escalate differs from gold")
        return round(score, 3), notes
    if role == "classify-issue":
        ok = out.get("issue_type") == gold.get("issue_type")
        notes.append("issue_type " + ("matches" if ok else f"differs ({out.get('issue_type')} vs {gold.get('issue_type')})"))
        return (1.0 if ok else 0.0), notes
    # structured-output (and any other json role): validate against gold's OWN
    # keys + value types (heterogeneous per-record schema).
    gold_keys = set(gold.keys())
    if not gold_keys:
        return 1.0, notes
    present = 0
    type_ok = 0
    for k, gv in gold.items():
        if k in out:
            present += 1
            if type(out[k]) is type(gv):
                type_ok += 1
    frac_present = present / len(gold_keys)
    frac_type = type_ok / len(gold_keys)
    notes.append(f"{present}/{len(gold_keys)} gold keys present, {type_ok} type-correct")
    return round(0.5 * frac_present + 0.5 * frac_type, 3), notes


def score_json(role: str, output: str, gold_obj: dict, schema: Optional[dict]) -> ScoreResult:
    res = ScoreResult(role=role, output_type="json", score=0.0)
    out = parse_json_loose(output)
    res.parses = isinstance(out, dict)
    if not res.parses:
        res.notes.append("does not parse as JSON object")
        return res
    # required keys: prefer per-record gold keys for heterogeneous roles,
    # else the manifest schema.
    required = set((schema or {}).get("required_keys", [])) or set(gold_obj.keys())
    res.keys_ok = required.issubset(out.keys())
    if not res.keys_ok:
        res.notes.append(f"missing keys: {sorted(required - set(out.keys()))}")
    # enum validity from manifest schema (fixed-contract roles only)
    enums = (schema or {}).get("enums", {}) if schema else {}
    enums_ok = True
    for k, allowed in enums.items():
        if allowed and k in out and out[k] is not None and out[k] not in allowed:
            enums_ok = False
            res.notes.append(f"{k}={out[k]!r} not in {allowed}")
    res.enums_ok = enums_ok
    res.match, mnotes = _match_json(role, out, gold_obj)
    res.notes += mnotes
    res.score = round(0.3 * res.keys_ok + 0.2 * res.enums_ok + 0.5 * res.match, 3)
    return res


# ---- Markdown scoring -----------------------------------------------------

_HEADER = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.MULTILINE)


def required_sections(gold: str) -> list[str]:
    """Section titles (markdown headers) present in the gold output."""
    return [h.strip().lower() for h in _HEADER.findall(gold)]


def markdown_gate(output: str, gold: str, system_prompt: str = "") -> tuple[float, list]:
    notes = []
    if not output or not output.strip():
        return 0.0, ["empty output"]
    # 1) section coverage
    req = required_sections(gold)
    out_headers = set(required_sections(output))
    if req:
        covered = sum(1 for h in req if h in out_headers)
        coverage = covered / len(req)
        notes.append(f"{covered}/{len(req)} gold sections present")
    else:
        coverage = 1.0  # gold had no headers; don't penalize
    # 2) length sanity vs gold (too short = likely truncated/empty-ish)
    ratio = len(output) / max(1, len(gold))
    length_ok = ratio >= 0.25
    if not length_ok:
        notes.append(f"output is {ratio:.0%} of gold length (suspiciously short)")
    # 3) prompt leakage: model echoing its system prompt back
    leak = False
    if system_prompt:
        probe = system_prompt.strip()[:60]
        if probe and probe in output:
            leak = True
            notes.append("leaks system prompt verbatim")
    gate = coverage * (1.0 if length_ok else 0.5) * (0.7 if leak else 1.0)
    return round(gate, 3), notes


def score_markdown(role: str, output: str, gold: str, system_prompt: str = "",
                   judge: Optional[Callable[[str, str, str], dict]] = None) -> ScoreResult:
    res = ScoreResult(role=role, output_type="markdown", score=0.0)
    res.gate, notes = markdown_gate(output, gold, system_prompt)
    res.notes += notes
    if judge is not None and res.gate > 0:
        jr = judge(role, output, gold)  # expects {faithfulness,completeness,grounding,format} 1-5
        vals = [jr.get(k, 0) for k in ("faithfulness", "completeness", "grounding", "format")]
        res.judge = round(sum(vals) / (5 * len(vals)), 3) if vals else None
        res.score = round(0.4 * res.gate + 0.6 * (res.judge or 0), 3)
        res.notes.append(f"judge raw={jr}")
    else:
        # deterministic-only: gate IS the score (judge wired later)
        res.score = res.gate
    return res


# ---- dispatch -------------------------------------------------------------

def tiered_score(record: dict, output: str, manifest: dict,
                 judge: Optional[Callable] = None) -> ScoreResult:
    """record: an eval/<role>.jsonl record. manifest: roles.json entry map."""
    role = record["role"]
    rm = manifest.get(role, {})
    out_type = rm.get("output_type", "markdown")
    if out_type == "json":
        gold_obj = parse_json_loose(record["gold"]) or {}
        return score_json(role, output, gold_obj, rm.get("schema"))
    return score_markdown(role, output, record["gold"], record.get("system", ""), judge)
