# archon-models

Local specialist models for [Archon](https://github.com/coleam00/Archon) — the open-source harness builder for AI coding agents. LoRA adapters fine-tuned on `unsloth/gemma-4-12b`, trained locally with Unsloth + TRL. Two lanes:

1. **v1 — the router**: natural-language request → the single best Archon **workflow** (or abstain + escalate). Small, focused, JSON-out.
2. **v2 — the execution model** *(new)*: the model produces the Archon **artifacts themselves** — implementation plans, issue investigations, code reviews, classifications — across **18 roles**, the way the frontier teacher does, but locally. **[Read the v2 training write-up →](docs/V2_TRAINING.md)**

> **Status:** router v0.1 → v0.1.1 (data drive below). **Execution model v2: trained 2026-06-10** (held-out eval_loss 0.698 → 0.631 over 3 epochs, no overfit); per-role evaluation via [PromptLab](promptlab/DESIGN.md) is in progress — see **[docs/TESTING.md](docs/TESTING.md)** for results and **[reports/](reports/)** to contribute your own testing data.

---

## Why this exists

Archon defines development processes (plan → implement → validate → review → PR) as YAML workflows. As the number of workflows grows, *choosing* the right one for a request becomes its own task. This repo is an experiment in training a **narrow, deep, local** specialist to do that routing cheaply and offline, instead of spending a frontier model's context on it.

## What's in here

```
scripts/                         # the regenerable data + training pipeline
  extract_routing.py             #   YAML workflows/commands -> intent→label anchor pairs
  build_routing_dataset.py       #   anchors + abstention -> train/eval JSONL (Gemma chat format)
  train_routing_local.py         #   Unsloth + TRL LoRA SFT on Gemma-4-12B
data/
  build/                         # the v1 training set (train.jsonl / eval.jsonl + stats)
  seeds/phase0_raw_extracts_v2/  # raw extracted anchors (workflow / command / subagent routing)
model/
  archon-router-gemma4-lora/     # the trained LoRA adapter (final, no checkpoints) — Git LFS
examples/
  infer_router.py                # load base + adapter, route a request
```

## How you can help build v0.1.1

**The model is only as good as the requests it learns from — and right now it's
only seen workflow descriptions, not how real people actually ask.** That's the
single biggest gap, and it's one anyone can help close without a GPU or any ML
work. If you've used Archon (or anything like it), you already know how you'd
phrase these requests. That's exactly the data we need.

**The goal of v0.1.1:** replace the canonical-anchor memorization with a corpus
of realistic, diverse, correctly-labeled requests so the router generalizes to
how people actually talk.

**What to contribute** (pick whichever you can; all are valuable):

| Priority | Bucket | What we need | Example |
|---|---|---|---|
| 🥇 Highest | **B — realistic paraphrases** | Natural ways to ask for a workflow — *not* echoing its description. ~15–30 varied phrasings per workflow. | "the login page bounces users back to sign-in, can you fix issue #412" → `archon-fix-github-issue` |
| 🥈 High | **C — boundary pairs** | Two near-identical requests that route to *different* workflows. These teach the hardest calls. | "just fix #500" → `archon-fix-github-issue` vs. "full fix+review pipeline on #500" → `archon-issue-review-full` |
| 🥉 High | **D — honest abstentions** | Plausible coding-adjacent requests that *no* workflow fits, so the model should escalate. | "should we rewrite the backend in Rust?" → `{escalate: true}` |
| Bonus | **A — real run history** | Actual `request → workflow that ran` pairs from your own usage (anonymized). | (from your Archon logs) |

**Two ways in:**

1. **Add data** — drop a JSONL file in [`data/contrib/`](data/contrib/), validate,
   open a PR. Start from the [**gold reference set**](data/contrib/_reference/archon-router-gold.jsonl)
   (39 curated examples) — copy its *style*, not its sentences.
2. **Just have an idea?** Open a
   [**Data idea** issue](../../issues/new?template=data-idea.md) — a workflow the
   router gets wrong, a confusable pair you've hit, a phrasing it should know. No
   code required.

Full walkthrough, do's and don'ts, and the schema:
**[CONTRIBUTING.md](CONTRIBUTING.md)** · **[docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md)**

The 30 routable workflow labels live in `.archon/workflows/*.yaml` and are listed
in `data/build/build_stats.json`. When in doubt about a label, use bucket **D**
(escalate) or open an idea issue rather than guessing.

## The routing task

**Input:** a user request (natural language).
**Output:** a JSON decision:

```json
{"workflow": "archon-fix-github-issue", "confidence": "high", "escalate": false}
```

When no workflow fits, the model abstains: `{"workflow": null, "confidence": "low", "escalate": true}`.

The label space is the set of routable Archon workflows (`.archon/workflows/*.yaml`, excluding `e2e-*` smoke tests) — 30 routable labels in v1 (see `data/build/build_stats.json`).

## Quick start (inference)

You need the Gemma base model (downloaded separately; accept the [Gemma license](https://ai.google.dev/gemma/terms)) plus this adapter.

```bash
pip install "transformers>=4.45" peft torch
git lfs install && git clone https://github.com/jmm2020/archon-models
python examples/infer_router.py "fix github issue #412 in my repo"
```

See `examples/infer_router.py` for the full load-and-route snippet.

## Reproduce the dataset + training

```bash
# 1. Extract intent→workflow anchors from live Archon YAML (point at your .archon/)
python scripts/extract_routing.py

# 2. Build the train/eval JSONL (Gemma chat format, stratified split + abstention)
python scripts/build_routing_dataset.py

# 3. Train the LoRA on Gemma-4-12B (Unsloth + TRL; --smoke for a 5-step dry run)
python scripts/train_routing_local.py            # full run
python scripts/train_routing_local.py --smoke    # validate the pipeline
```

> The scripts assume a local Archon install (`.archon/workflows/`, `.archon/commands/`). Adjust the paths near the top of each script to your layout.

## Data strategy

v1 is trained on canonical anchors: each workflow's own `Use when:`, `Triggers:`, and `NOT for:` text, plus a small hardcoded abstention set — roughly 200 examples total.

That is enough to validate the training pipeline, but it is not enough to prove routing skill. Training and evaluating on canonical anchors mostly measures memorization. Real users do not phrase requests like workflow descriptions.

The path to a genuinely useful router is:

1. **Real run history**
   Mine actual `user request → workflow that ran` pairs. These are perfectly labeled and reflect the real distribution. They are the highest-value examples, but they are usually thin and template-skewed, so they should guide label priority rather than serve as the whole training set.

2. **Synthesized realistic paraphrases**
   Generate 15–30 diverse, natural user phrasings per workflow from each workflow's `Use when`, `Does`, and `NOT for` descriptions. This is the biggest lever. The `synth/` bucket in `build_routing_dataset.py` is the hook for this.

3. **Contrastive boundary pairs**
   Every `NOT for: ... use X` clause identifies a confusable sibling workflow. Turn those into hard A-vs-B examples. These teach the exact boundaries a multi-class router is most likely to get wrong.

4. **Honest abstention examples**
   Include coding-adjacent near misses, vague requests, and genuinely ambiguous cases that should escalate instead of route. Avoid toy out-of-domain trivia.

5. **A thin retention slice**
   Keep a small slice of general examples so the base model does not collapse into a JSON-only router.

Two principles matter most:

* Keep the dataset regenerable and keyed to the current workflows, so it rebuilds cleanly when workflows change. Freeze nothing.
* Evaluate on real or diverse held-out user phrasings, never on the same canonical anchors used for training.

## Model details

| | |
|---|---|
| Base | `unsloth/gemma-4-12b` |
| Method | LoRA SFT (PEFT) via Unsloth + TRL |
| LoRA | r=32, α=32; targets q/k/v/o + gate/up/down proj |
| Precision | 4-bit base, bf16 train |
| Adapter | `model/archon-router-gemma4-lora/` (~525 MB, Git LFS) |

See `model/archon-router-gemma4-lora/MODEL_CARD.md` for the auto-generated card and framework versions.

## License

- Code, scripts, datasets, and docs: **MIT** (see `LICENSE`).
- The LoRA adapter derives from Gemma and is additionally subject to the **[Gemma Terms of Use](https://ai.google.dev/gemma/terms)** — see `NOTICE`.
