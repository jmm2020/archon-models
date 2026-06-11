# Testing Gemma4-Archon — PromptLab + the reports area

This is the **reporting and testing area** for the working group: how we
measure the model, how to run the harness yourself, and where results live.

## How we evaluate (PromptLab)

[`promptlab/`](../promptlab/) is a per-role **prompt generate-and-evaluate
harness**: hold the trained model fixed, vary the per-role system prompt, score
the outputs objectively against the held-out gold set.

```
baseline prompt (per role) ┐
role_spec.md               ├─► [Author: frontier] ─► K candidate prompts / role
                           ┘            │
                 eval set (165, grouped by role)
                            │
   per role · per candidate · per input:
      [Runner: Gemma4 + v2 adapter, temp 0] ─► output
      [Tiered scorer]  JSON roles: parse/schema/enum/field-match (deterministic)
                       markdown roles: section-coverage gate + frontier judge
                            │
      [Report] ─► per-role leaderboard · winner · Δ vs baseline
```

Design rationale and the full component table: [`promptlab/DESIGN.md`](../promptlab/DESIGN.md).

Two properties we hold ourselves to:

- **Deterministic where possible.** JSON roles (routing, classify-issue,
  escalation, structured-output) are scored with zero LLM judgment — parse,
  schema, enum, field match vs gold.
- **Honest where not.** Markdown roles get a deterministic gate (required
  sections, length sanity, prompt-leak detection) plus a frontier judge rubric
  (faithfulness / completeness / grounding / format, 1–5). Judge inputs are
  blinded to which prompt variant produced them.

## Run it yourself

```bash
cd archon-models
pip install -r requirements.txt   # plus unsloth for the GPU runner

# unit tests (no GPU, no network)
python -m pytest promptlab/ -q

# score an existing run + rebuild the leaderboard (no GPU)
python promptlab/score_run.py --run subset_v2
python promptlab/report.py --scores promptlab/runs/promptlab_subset_v2/scores.jsonl --run subset_v2

# full generation run (needs the GPU + the v2 adapter from Releases)
CUDA_VISIBLE_DEVICES=0 python promptlab/runner.py --jobs promptlab/jobs.jsonl --run my_run
```

## Results

### Subset proof (create-plan · classify-issue · routing) — 2026-06-11

Run `subset_v2_gguf`: 100 generations (25 records × baseline + 3 candidates),
v2 q8_0 GGUF via llama-server, temp 0, 16.6s/gen avg on one RTX 3090.
**Deterministic tier** (frontier judge for create-plan prose quality pending):

| Role | baseline | best candidate | Δ | n |
|---|---|---|---|---|
| classify-issue | **1.000** | 1.000 (all three) | 0.0 | 11 |
| create-plan (gate) | **0.951** | 0.944 (`v1_phase_contract_explicit`) | −0.007 | 11 |
| routing ⚠️ thin | **0.767** | 0.650 (`v1`/`v3`) | −0.117 | 3 |

**Headline finding: the trained baseline prompt won all three roles.** The
fine-tuned model is anchored to its training-prompt distribution — every
"improved" candidate scored equal or worse. Practical guidance: **deploy the
training prompts verbatim**; invest prompt-engineering effort in the corpus,
not at inference time. (classify-issue went 11/11 on every variant — the model
is robust on that task regardless of prompt wording.)

The subset proves the methodology on one markdown role + two JSON roles before
we scale to all 18. routing is flagged **low-confidence** (only 3 eval records)
— treat its numbers as directional.

## Reporting your own testing data (`reports/`)

We want the group's eyes on this model. File what you find:

1. **Structured test reports** → PR a markdown file into
   [`reports/`](../reports/) (template in `reports/README.md`): what you ran,
   on what hardware, inputs, outputs, what was good, what was wrong.
2. **One-off findings** (a bad generation, a role that drifts, a prompt that
   works better) → open an issue with the `model-feedback` label and include
   the role, the input, and the output verbatim.
3. **Eval data** — new held-out records for thin roles (routing, escalation,
   structured-output, memory) are the highest-value contribution. Same JSONL
   shape as `promptlab/eval/<role>.jsonl`: `{"role", "scenario", "system",
   "user", "gold"}`.

Anonymize anything from real repos before posting.
