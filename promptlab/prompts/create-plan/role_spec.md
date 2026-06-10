# Role spec: create-plan

**What the model must do:** Act as an implementation *planner* for a given
codebase (stack named in the scenario context). Given a feature description or a
path to a PRD, produce a battle-tested, codebase-grounded implementation plan.

**Output:** A single `plan.md` markdown artifact. **Writes zero code.** Operates
in strict phase order (detect input type → explore → plan).

**Output shape (markdown sections that should appear):**
- Feature title + **Summary** + **User Story**
- Concrete, ordered **Tasks** (file-by-file, each independently verifiable)
- **Risks** / edge cases
- Grounding in the named stack (real file/dir/library references, not generic advice)

**Success criteria (what a good prompt makes the model do better):**
1. **Faithfulness** — the plan addresses the actual requested feature, no drift.
2. **Completeness** — covers data model, API, UI, tests, and edge cases as applicable.
3. **Grounding** — references the scenario's real stack/conventions, not generic boilerplate.
4. **Format adherence** — clean `plan.md` structure, no code, no working narration.

**Failure modes to penalize:** writing code; vague non-grounded tasks; skipping
phases; leaking/echoing the system prompt; truncated or stub plans.

**Prompt-engineering constraint:** the model is fine-tuned on the v2 create-plan
prompts — candidates should refine structure/explicitness within that distribution
(e.g. clearer phase contract, explicit required sections), not impose a foreign format.
