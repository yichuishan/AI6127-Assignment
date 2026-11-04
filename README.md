# AI6127-Assignment

Evidence-grounded RAG summarization and generation experiments.

## Run the experiment webpage

Requires Python 3.10+. Download this branch with **Code → Download ZIP**, extract
it, open a terminal in the extracted folder, and run:

```bash
python launch_web.py
```

The browser opens automatically. Enter your OpenAI-compatible API base URL, model
ID and API key in the page, then start with three queries. Basic generation and
the webpage use only Python's standard library. Keys are held in memory and are
not saved to experiment files or browser storage. Stop the terminal with Ctrl+C
when finished.

The page shows progress, API calls, failures and summary previews, and exports a
ZIP containing outputs, citations, verification audits, CSV metrics and a report.
A separate judge model enables final faithfulness/relevance estimates.

## Included data

`experiments/steam_generation_eval.jsonl` contains 100 real query cases, each with
up to ten saved `hybrid_rerank` evidence chunks, from the pinned teammate repository.
These are condensed annotated-benchmark rankings, not production top-k results.
Human summary references and key points have not yet been added; those are
required for reference metrics and information coverage. This checkout does not
include the complete retrieval corpus, indexes or all retrieval benchmark runs.

For optional ROUGE/BERTScore dependencies:

```bash
python -m pip install -r requirements-generation-eval.txt
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for experiment commands, metrics, limitations
and private key setup, and [README_generation.md](README_generation.md) for the
generator interface and verification methods.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use scripted clients; passing tests do not establish real LLM quality gains.
