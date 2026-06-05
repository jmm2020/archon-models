# data/contrib

Community-contributed training examples live here. **Add your examples as JSONL
files in this directory** (one example per line), then open a pull request.

- Format & rules: [`../../docs/DATA_SCHEMA.md`](../../docs/DATA_SCHEMA.md)
- How to contribute: [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md)
- Working template: [`EXAMPLE.jsonl`](EXAMPLE.jsonl)

Suggested layout — one folder per contributor keeps PRs clean:

```
data/contrib/
  EXAMPLE.jsonl              # reference template (don't train on this)
  <your-github-handle>/
    fix-issue-paraphrases.jsonl
    pr-review-boundaries.jsonl
```

Validate before you push:

```bash
python scripts/validate_examples.py data/contrib
```

> `EXAMPLE.jsonl` is a template for humans, not training data — the build
> pipeline excludes it.
