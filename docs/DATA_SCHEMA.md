# Data schema (contribution format v0.1)

Every contributed training example is **one JSON object per line** (JSONL). This
is the human-facing contribution format — it carries provenance (`meta`) so we
know where each example came from. The build pipeline maps it into the internal
training format (`conversations`) at dataset-build time; contributors never need
to touch the internal format.

`scripts/validate_examples.py` enforces this schema and runs on every pull
request. Run it locally before opening a PR:

```bash
python scripts/validate_examples.py data/contrib
```

## Record shape

```json
{
  "messages": [
    {"role": "system", "content": "You are the Archon workflow router. ..."},
    {"role": "user", "content": "fix github issue #412, the login redirect loop"},
    {"role": "assistant", "content": "{\"workflow\": \"archon-fix-github-issue\", \"confidence\": \"high\", \"escalate\": false}"}
  ],
  "meta": {
    "bucket": "B",
    "author": "your-github-handle",
    "source": "synthetic",
    "note": "optional free-text — why this example exists / what it teaches"
  }
}
```

### `messages` (required)
- A list of **2 or more** turns.
- Each turn is `{"role": ..., "content": ...}`.
- `role` ∈ `system`, `user`, `assistant`, `tool`.
- `content` is a non-empty string.
- Must include at least one `user` turn and one `assistant` turn.
- The **final turn must be `assistant`** (the target the model learns to produce).

For the routing task, the assistant's content is the decision JSON:

```json
{"workflow": "<name or null>", "confidence": "high"|"medium"|"low", "escalate": true|false}
```

When nothing fits, abstain and escalate:
`{"workflow": null, "confidence": "low", "escalate": true}`.

### `meta` (required)
| Field | Required | Values | Meaning |
|---|---|---|---|
| `bucket` | yes | `A`–`H` | Which data bucket this example belongs to (see below) |
| `author` | yes | your GitHub handle | Who contributed it (for attribution + accountability) |
| `source` | no | `synthetic`, `distilled`, `real-run`, `hand-written` | How it was produced |
| `note` | no | free text | Optional — why it exists / what boundary it teaches |

## Buckets

The router's quality is bounded by how well these buckets are filled. The two
biggest levers are **B** (realistic paraphrases) and **C** (boundary pairs).

| Bucket | Name | What it is | Notes |
|---|---|---|---|
| **A** | Real run history | Actual `user request → workflow that ran` pairs | Perfectly labeled; reflects real distribution. Highest value but usually thin/template-skewed — use to guide label priority. |
| **B** | Realistic paraphrases | Diverse, natural user phrasings per workflow | **Biggest lever.** Aim for 15–30 varied phrasings per workflow. Don't echo the workflow's own description. |
| **C** | Contrastive boundary pairs | Hard A-vs-B examples from `NOT for: ... use X` clauses | Teaches the exact confusable-sibling boundaries a router gets wrong. |
| **D** | Honest abstention / escalation | Near-misses, vague, ambiguous requests that should escalate | Use coding-adjacent misses, not toy out-of-domain trivia. |
| **E** | Agentic execution traces | Multi-step tool-use sequences inside a workflow | For the broader frontier-substitute scope, not router-only. |
| **F** | Memory-domain operations | `create_memory` / search / recall interaction patterns | Teaches the memory-domain reflexes. |
| **G** | PLAN/EXEC escalation boundary | When to handle locally vs escalate hard planning/design to a frontier model | The trained escalation reflex — the core of the vision. |
| **H** | Retention slice | Small slice of general examples | Keeps the base model from collapsing into a JSON-only router. |

> v1 of the router only uses buckets A–D (it's routing-only). E–H exist for the
> broader Gemma4-Archon frontier-substitute scope. If you're contributing router
> data, you'll almost always be in **B**, **C**, or **D**.

## Two rules that matter most

1. **Don't train on the canonical anchors.** Real users don't phrase requests
   like workflow descriptions. Paraphrase naturally; vary length, tone, and
   specificity.
2. **No secrets, no PII.** The validator scans for API keys, tokens, absolute
   home paths, and personal emails — but scrub before you commit. Anonymize repo
   names and issue text if they're private.

See `CONTRIBUTING.md` for the full pull-request flow.
