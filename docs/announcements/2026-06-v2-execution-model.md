# Gemma4-Archon v2: a local execution model for Archon — trained, evaluated, and ready for your testing

Hey all — big update on the local-models lane.

## TL;DR

We trained **v2**: a LoRA on Gemma-4-12B that doesn't just *route* Archon
requests (v1) but **produces the execution artifacts themselves** —
implementation plans, issue investigations, code reviews, classifications —
across **18 Archon roles**, locally, on one consumer GPU. Held-out eval_loss
went 0.698 → 0.645 → **0.631** over 3 epochs with no overfit signature, and we
built an objective per-role evaluation harness (**PromptLab**) to measure
prompt quality and the gap to the frontier teacher honestly.

- 📖 **What we did + why:** [docs/V2_TRAINING.md](../V2_TRAINING.md)
- 🧪 **How we test + results:** [docs/TESTING.md](../TESTING.md)
- 📋 **Report your own testing:** [reports/](../../reports/)
- 📦 **The v2 adapter:** attached to the release (see Releases)

## What we did (short version)

1. **Corpus:** 2,920 train / 165 held-out eval records distilled from an Opus
   teacher over scenario-grounded synthetic codebases (each with a real stack +
   repo conventions, so outputs must be specific, not generic). 18 roles: 14
   markdown-artifact roles (~195 records each) + 4 strict-JSON roles + memory.
2. **Training:** responses-only LoRA SFT (r=32) on `unsloth/gemma-4-12b` 4-bit,
   max_seq **4096** — auditing token lengths first mattered: at 2048, 39% of
   records would have truncated mid-plan. 3 epochs / 549 steps / ~7.6 h on a
   single RTX 3090.
3. **Evaluation:** PromptLab — hold the model fixed, vary the per-role system
   prompt, score objectively (deterministic parse/schema/field checks for JSON
   roles; section-coverage gate + blinded frontier-judge rubric for markdown
   roles). First results from the 3-role subset proof are in
   [docs/TESTING.md](../TESTING.md).

## Why we think this matters

Most Archon run tokens go to *standard-shaped artifact production*, not novel
reasoning. A local specialist that handles that 80% means: frontier budget
reserved for the hard calls, fully-offline pipelines for private repos, and
prompt/eval iteration cheap enough to run constantly. And the recipe
(scenario-grounded distillation + responses-only LoRA + honest held-out eval +
objective prompt evaluation) is reusable for other harnesses the group cares
about.

## What we'd like from the group

1. **Run it.** Grab the adapter from Releases, point it at the eval set or your
   own (anonymized) missions, and file what you find in
   [reports/](../../reports/) (template inside) or a `model-feedback` issue.
2. **Eval data for the thin roles.** routing (57 train / 3 eval), escalation
   (47/3), structured-output (47/3), memory (38/2) — even a handful of
   well-labeled records each materially improves what we can claim.
3. **Real usage pairs.** Anonymized `mission → artifact` pairs from actual
   Archon runs are the single biggest lever for v3.
4. **Skepticism.** The methodology docs are linked above — poke holes in the
   scoring, the corpus design, the claims. That's what this thread is for.

Questions, ideas, and "this output is garbage, here's why" all welcome below. 🚀
