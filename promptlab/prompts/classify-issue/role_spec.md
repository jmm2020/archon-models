# Role spec: classify-issue

**What the model must do:** Given a GitHub issue (title, body, labels, comments,
state, url, author) as JSON, determine its type from a fixed enum and explain why.

**Output:** A single JSON object, no other text:
```json
{"issue_type": "bug|feature|enhancement|refactor|chore|documentation",
 "title": "<concise title>",
 "reasoning": "<clear, signal-grounded justification>"}
```

**Contract (hard, checked deterministically):**
- Parses as a JSON object.
- Keys present: `issue_type`, `title`, `reasoning`.
- `issue_type` ∈ the 6-value enum.

**Success criteria:**
1. **Correctness** — `issue_type` matches the gold label (the primary signal).
2. **Reasoning quality** — cites concrete signals (repro steps, expected/actual,
   feature-request language) rather than restating the title.
3. **Format** — JSON only, no markdown fence or prose around it.

**Failure modes to penalize:** wrong `issue_type`; missing keys; non-JSON output;
reasoning that just echoes the issue body; hedging across multiple types.

**Prompt-engineering constraint:** stay near the trained prompt; the lever here is
sharpening the enum definitions and the "JSON only" instruction so the model stops
emitting fences or trailing commentary.
