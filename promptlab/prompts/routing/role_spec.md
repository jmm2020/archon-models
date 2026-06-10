# Role spec: routing

**What the model must do:** Given a natural-language user request, select the
single best Archon workflow to run, or abstain and escalate when none fits.

**Output:** A single JSON object, no other text:
```json
{"workflow": "<name or null>", "confidence": "high|med|low", "escalate": true|false}
```
Abstain form when nothing fits: `{"workflow": null, "confidence": "low", "escalate": true}`.

**Contract (hard, checked deterministically):**
- Parses as a JSON object.
- Keys present: `workflow`, `confidence`, `escalate`.
- `confidence` ∈ {high, med, low}.

**Success criteria (scored vs gold):**
1. **Workflow correctness** — picks the gold workflow (primary signal, 0.7 weight).
2. **Escalation correctness** — `escalate` matches the gold decision (0.3 weight).
3. **Calibration** — `confidence` is high only when the route is unambiguous.
4. **Format** — JSON only.

**Failure modes to penalize:** wrong workflow; routing a request that should
escalate (and vice-versa); inventing workflow names outside the label set;
non-JSON output.

**Prompt-engineering constraint:** the model learned a fixed workflow label set —
candidates should reinforce abstention discipline and confidence calibration, not
expand or rename the label space.
