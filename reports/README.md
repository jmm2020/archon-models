# Test reports

Community testing data for Gemma4-Archon. One markdown file per report, named
`YYYY-MM-DD-<github-handle>-<topic>.md`. PRs welcome — see
[docs/TESTING.md](../docs/TESTING.md) for what belongs here vs. an issue.

## Template

```markdown
# <short title>

- **Date / author:** 2026-06-15 / @yourhandle
- **Adapter:** v2 (release tag) · **Base:** unsloth/gemma-4-12b 4-bit
- **Hardware:** e.g. RTX 3090 24GB
- **Roles tested:** create-plan, routing, ...
- **Inputs:** synthetic / own repo (anonymized) / promptlab eval set

## What I ran
(commands or harness run label)

## Results
(scores, timings, or qualitative observations — include verbatim outputs for
anything surprising, good or bad)

## Issues found
(wrong outputs, format breaks, truncations — link issues you opened)

## Suggestions
(prompt variants that worked better, data gaps, harness improvements)
```
