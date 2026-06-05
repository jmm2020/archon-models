# Contributing to archon-models

Thanks for helping make the router better. The single highest-impact thing you
can do is **add training examples** — realistic user phrasings and hard boundary
cases. You can also just **propose ideas** without writing data (see below).

## Two ways to contribute

### 1. Add training data (a pull request)

1. **Fork & branch.**
   ```bash
   git clone https://github.com/<you>/archon-models
   cd archon-models
   git checkout -b data/<short-topic>
   ```

2. **Add a JSONL file under `data/contrib/`.** One example per line. Put it in a
   file named for what it covers, e.g.
   `data/contrib/<your-handle>/fix-issue-paraphrases.jsonl`.
   Each line follows [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md):
   ```json
   {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "{\"workflow\": \"...\", \"confidence\": \"high\", \"escalate\": false}"}], "meta": {"bucket": "B", "author": "your-handle", "source": "synthetic"}}
   ```
   See `data/contrib/EXAMPLE.jsonl` for a working template you can copy.

3. **Validate locally** (this is the same check CI runs):
   ```bash
   python scripts/validate_examples.py data/contrib
   ```
   Fix anything it flags. Green = ready.

4. **Open a PR.** The CI gate re-runs the validator on your files. A maintainer
   reviews for realism and label correctness, then merges. Your examples get
   picked up the next time the dataset is rebuilt.

#### What good data looks like
- **Realistic phrasings** (bucket B): how a real person would actually ask —
  not a paraphrase of the workflow's own description.
- **Boundary pairs** (bucket C): two near-identical requests that route to
  *different* workflows. These teach the hardest distinctions.
- **Honest abstentions** (bucket D): plausible coding-adjacent requests that
  *no* workflow fits, so the model should escalate.
- Aim for **variety over volume**. Ten diverse examples beat fifty rephrasings
  of the same sentence.

#### What gets a PR bounced
- Secrets, tokens, personal emails, or absolute `/home/...` paths (the validator
  catches these, but scrub first).
- Examples copied verbatim from workflow descriptions (memorization, not skill).
- Wrong or guessed labels. If you're unsure which workflow a request maps to,
  use bucket **D** (escalate) or open an idea issue instead.

### 2. Propose an idea (an issue)

No data, no code — just a suggestion. Open a
[**Data idea** issue](../../issues/new?template=data-idea.md): a workflow that's
under-covered, a confusable pair you've hit, a phrasing the router gets wrong, or
a whole bucket worth filling. Maintainers (or other contributors) can turn it
into data.

## Ground rules
- **Code, data, and docs are MIT** (`LICENSE`). By contributing you agree your
  contribution is released under MIT.
- The adapter additionally inherits the **[Gemma Terms of Use](https://ai.google.dev/gemma/terms)** (`NOTICE`).
- Be kind and specific in reviews. The goal is a genuinely useful router, so
  realism and correctness beat volume every time.
