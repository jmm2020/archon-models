# Gemma4-Archon v2 — the execution model

**Status: v2 adapter trained (2026-06-10). Evaluation in progress — see [TESTING.md](TESTING.md).**

v1 taught a local model to *route* (request → workflow name). v2 is the second,
much broader lane: teaching the same base model to **produce the execution
artifacts themselves** — implementation plans, issue investigations, code
reviews, fix traces — the way a frontier teacher model does, but locally, on a
single consumer GPU, with no API cost per call.

This doc is the context for the working group: what we built, how the training
went, what we think it's good for, and where we'd like to take it together.

---

## What we did

### The corpus

2,920 training / 165 held-out eval records across **18 Archon roles**, distilled
from an Opus teacher over synthetic but fully-grounded codebase scenarios
(each scenario is a coherent fictional product — name, stack, repo conventions —
so outputs must ground in *specific* technical context, not generic advice).

| Role family | Roles | Output | ~Records each |
|---|---|---|---|
| Planning / execution | create-plan, implement-tasks, investigate-issue, fix-issue, self-fix-all, simplify-changes, validate | markdown artifact | 195 |
| Review | code-review, error-handling-review, test-coverage-review, comment-quality-review, docs-impact-review, synthesize-review | markdown artifact | 195 |
| Decision (JSON) | routing, escalation, classify-issue, structured-output | strict JSON | 47–195 |
| Memory | memory | prose | 38 |

Records are full chat conversations (system prompt defining the role +
scenario, user mission, teacher's gold response) in Gemma chat format.

### The training run

| | |
|---|---|
| Base | `unsloth/gemma-4-12b`, 4-bit |
| Method | LoRA SFT (r=32, α=32, q/k/v/o + gate/up/down proj), responses-only loss |
| Sequence length | **4096** (see below — this one mattered) |
| Schedule | 3 epochs, 549 steps, effective batch 16, lr 2e-4 linear, bf16 |
| Hardware | single RTX 3090 (24 GB), ~7.6 h wall clock |
| Script | [`scripts/train_execution_local.py`](../scripts/train_execution_local.py) |

**Loss trajectory** (held-out eval, the only honest signal):

| Checkpoint | eval_loss |
|---|---|
| pre-train smoke | 1.431 |
| epoch 1 | 0.6983 |
| epoch 2 | 0.6447 |
| epoch 3 (final) | **0.6314** |

Eval tracked train loss (final train 0.6714) the whole way — the model is
generalizing to held-out scenarios, not memorizing. No divergence, no overfit
signature at 3 epochs.

### The lesson that almost bit us: sequence length

v1 trained at `max_seq=2048` and that was fine for routing JSON. The v2 corpus
has p95 ≈ 3.3k and p99 ≈ 3.9k tokens — at 2048, **39% of records would have
truncated mid-plan**, silently cutting off the exact completions we train on
(responses-only loss makes this worse: the truncated tail *is* the target).
Audit your token distribution before you train. 4096 captures 99.6% of records
fully and still fits a 12B 4-bit LoRA on a 24 GB card at batch 1 × grad-accum 16.

---

## What it does

Given one of the 18 role system prompts and a task input, the adapter produces
the role's artifact: a `plan.md`, a structured code review, an issue
classification JSON, a routing decision. Same shape as the teacher's output,
produced locally at temp 0.

What we are explicitly **not** claiming yet: parity with the teacher. The eval
harness below exists to measure the gap honestly, role by role.

## Why we think this is worth pursuing

1. **Cost structure.** Archon runs burn most of their tokens on *artifact
   production* (plans, reviews, traces) — not on hard reasoning about novel
   problems. If a local 12B specialist handles the standard-shaped 80%, frontier
   calls reserve for the hard 20%.
2. **Privacy / offline.** Repos that can't leave the building can still get the
   full plan→review pipeline.
3. **Latency and iteration.** Local temp-0 generation makes prompt/eval loops
   (like PromptLab, below) cheap enough to run on every change.
4. **A reusable recipe.** Scenario-grounded distillation + responses-only LoRA +
   honest held-out eval is a pattern the group can re-apply to other harnesses,
   not just Archon.

## What we'd like to do next (and where you come in)

- **PromptLab eval (in progress):** per-role prompt engineering with objective
  scoring — deterministic gates for JSON roles, frontier judge for markdown
  roles. The subset proof (create-plan, classify-issue, routing) is running now;
  results land in [TESTING.md](TESTING.md) and `reports/`.
- **Scale the eval to all 18 roles** once the subset methodology proves out.
- **Thicken the thin roles.** routing (57), escalation (47), structured-output
  (47) and memory (38) need more training data — same contribution model as the
  v1 router data drive (see [CONTRIBUTING.md](../CONTRIBUTING.md)).
- **Real-usage data.** Synthetic scenarios got us off the ground; anonymized
  real `mission → artifact` pairs from your Archon runs are the next big lever.
- **Closed-loop prompt optimization** (PromptLab "Approach C"): wrap the
  generate→eval cycle in an optimizer once single-pass eval is trustworthy.
- **Test it on your repos and report back** — that's what [TESTING.md](TESTING.md)
  and the `reports/` area are for.

## Reproduce

```bash
# smoke (validates the whole pipeline in ~minutes, saves nothing)
CUDA_VISIBLE_DEVICES=0 python scripts/train_execution_local.py --smoke

# full run (~7.6 h on a 24 GB card)
CUDA_VISIBLE_DEVICES=0 python scripts/train_execution_local.py
```

Adapter lands in `scripts/runs/archon-gemma4-v2-lora/`. The trained v2 adapter
is published as a GitHub Release artifact (see the repo's Releases page).

## License

Same as the repo: code/data/docs MIT; the adapter derives from Gemma and is
additionally subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
