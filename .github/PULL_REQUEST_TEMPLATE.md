<!-- For data contributions. For code/docs, delete this and describe your change. -->

## What this adds
<!-- e.g. "25 realistic paraphrases for archon-fix-github-issue (bucket B)" -->

## Buckets covered
<!-- A=real-run  B=paraphrases  C=boundary pairs  D=abstention  (see docs/DATA_SCHEMA.md) -->
- [ ] A — real run history
- [ ] B — realistic paraphrases
- [ ] C — contrastive boundary pairs
- [ ] D — honest abstention / escalation

## Checklist
- [ ] Ran `python scripts/validate_examples.py data/contrib` locally — it passes
- [ ] No secrets, tokens, personal emails, or absolute `/home/...` paths
- [ ] Phrasings are realistic — not copied from workflow descriptions
- [ ] Labels are correct (or used bucket D / escalate where unsure)
- [ ] `meta.author` is my GitHub handle
