# PromptLab — per-role prompt generate-and-evaluate harness

**Status:** v0.1 design (Approach B). Date: 2026-06-10.

## Purpose

Engineer the **inference-time system prompt** the trained Gemma4-Archon model
uses for each Archon role, and measure prompt quality objectively. Hold the
model fixed, vary the per-role system prompt, score the model's output.

Key constraint: Gemma4-Archon is **fine-tuned** on a specific system-prompt
distribution (the v2 corpus). The win is finding the best prompt *within or
near* that distribution plus robustness — not a wholesale rewrite the model
never saw.

## The 18 roles (from the v2 corpus `meta.role`)

- **Markdown-output (14):** create-plan, implement-tasks, investigate-issue,
  fix-issue, code-review, error-handling-review, test-coverage-review,
  comment-quality-review, docs-impact-review, synthesize-review, self-fix-all,
  simplify-changes, validate, (memory — prose, thin).
- **JSON-output (4):** routing, escalation `{workflow, confidence, escalate}`;
  classify-issue `{issue_type∈6, title, reasoning}`; structured-output
  `{decision∈3, reason, required_actions[]}`.

## Approach B (chosen)

Frontier-authored candidates + single tiered-eval pass per invocation.
Re-invoke to iterate. Built so the closed-loop optimizer (Approach C) can later
wrap the generate→eval step. Generator + judge = **Opus 4.8** (temp 0 eval
decoding for reproducibility).

## Data flow

```
baseline prompt (per role) ┐
role_spec.md               ├─► [Author: Opus] ─► K candidate prompts / role
last run's failures (opt)  ┘                          │
                            eval set (165, grouped by role)
                                       │
  per role · per candidate · per input:
     [Runner: Gemma4-Archon base+adapter, temp 0] ─► output
     [Tiered scorer] det. gate (role-type aware) + Opus judge (markdown)
                                       │
     [Aggregator] ─► per-role leaderboard · winner · Δ vs baseline
```

## Components (each a small, testable unit)

| File | Job | GPU? | Frontier? |
|---|---|---|---|
| `extract_baselines.py` | per-role baseline prompt + `roles.json` manifest (output_type, contract, scenario-template structure) from `train.jsonl` | no | no |
| `build_eval_set.py` | group `eval.jsonl` by role → `eval/<role>.jsonl`; flag thin roles | no | no |
| `prompts/<role>/role_spec.md` | hand-seeded once: role job, output contract, success criteria | no | no |
| `author.py` | Opus generates K variants from baseline + spec (+ failures) | no | yes |
| `runner.py` | load model once; generate per (candidate × input); cache | **yes** | no |
| `scorer.py` | tiered: deterministic gate + Opus judge | no | judge only |
| `report.py` | per-role leaderboard, winner, Δ vs baseline, low-confidence flags | no | no |
| `promptlab.workflow.js` / `run_promptlab.py` | orchestrate author → run+score → report | mixed | yes |

## Scenario templating (empirically tested → revised)

System prompts embed scenario context ("…for the *mealwise* codebase (stack)").
A naive LCP/LCS factoring (`extract_baselines.py`) was tested and **FAILED**:
the invariant prefix+suffix span is only **1–16%** of each prompt across all 18
roles. The scenario is interpolated *throughout* (name + stack recur in many
places, phases differ), not in one clean middle slot. So string-op templating is
out.

**Revised plan:** factoring is a **frontier task** (`factor_prompts.py`, Opus):
given N sample prompts for a role, emit (1) an invariant role-instruction
template with an explicit `{{SCENARIO_CONTEXT}}` slot and (2) each record's
scenario_context. Validate by checking `baseline_template + scenario_context`
reproduces the original prompt. K candidate templates are then instantiated per
eval record by cheap string substitution.

Baseline score still uses each eval record's **own** verbatim system prompt
(already scenario-correct) — no factoring needed for the baseline line.

`extract_baselines.py` is retained as the **diagnostic + verbatim sample**
writer (`prompts/<role>/sample.txt`) and `roles.json` manifest source.

## `structured-output` is heterogeneous

It is not one schema — 150+ distinct keys across records (release gates, k8s
configs, ADRs, incident routing, …). Each record carries its **own** mini-schema.
Its scorer validates the model output against the **gold record's own keys +
value types**, not a fixed contract. (Fixed contracts apply to routing,
escalation, classify-issue.)

## Tiered scoring

- **JSON roles:** parse → schema/enum/field validation → exact-or-field match
  vs gold. Deterministic, no judge.
- **Markdown roles:** gate (required sections present, no prompt leakage, length
  sane) → Opus judge rubric: faithfulness-to-input, completeness, grounding,
  format-adherence (1–5 each) → composite. Optional pairwise vs gold.

## Error handling

Gen failure → null output, score 0, logged. Judge parse-fail → 1 retry →
deterministic-only + flag. Thin-eval roles (< ~5) → run but mark
low-confidence. Pinned judge model + temp 0 → reproducible.

## Testing

- Smoke: 1 role / baseline-only / 2 inputs against the **base model** (proves
  wiring before v2 adapter is ready).
- Unit: scorer on synthetic valid/invalid JSON + missing-section markdown →
  known scores.
- Golden: baseline prompt should roughly reproduce gold (sanity the trained
  model honors its training prompt).

## Scope lock (YAGNI)

v1 = Approach B, single round, Opus author + judge. **Prove on a subset first** —
`create-plan` (markdown) + `classify-issue` + `routing` (json) — then scale to
all 18. **Not building now:** closed-loop optimizer (C), judge panels, mutation
operators.
