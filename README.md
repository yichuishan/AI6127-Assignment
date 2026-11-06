# AI6127-Assignment

Evidence-grounded summarization with structured fact selection experiments.

## Run the webpage

Python 3.10+ is required. Download this branch, extract it, and run:

```bash
python launch_web.py
```

Enter **only a generation API URL, model ID, and key**. Start with three queries.
The page no longer configures or calls an evaluator model. Basic operation uses
Python's standard library; no scoring model or embedding download is required.
Keys remain in memory and are excluded from result files and browser storage.

## Current experiment

The generator returns a natural-language summary and per-claim `fact_ids`.
The program rebuilds `selected_fact_ids` from final retained claims and scores
these IDs against the annotated benchmark using set matching:

- Fact-selection Precision, Recall and F1.
- Game/aspect/sentiment coverage-unit recall.
- Positive/negative experience coverage for applicable contrast groups.
- Invalid or unannotated fact selections, latency, calls, tokens and optional cost.

The page shows progress, individual scores and a final method table. Downloaded
ZIPs contain predictions, reference sets, full outputs, checkpoints, CSV metrics
and a Markdown report. **The scores evaluate structured selection, not free-text
entailment.** Condition retention is not reported because conditions remain
unreviewed candidates.

A/B/C/D remain available. C/D use the generation model for their internal
verification/revision; final scoring always uses deterministic code. B/C/D share
an initial draft. Their call counts include this internal generation work.

To prevent gold-label leakage, the model sees every one of eight possible aspect
slots for each input review, including unmentioned aspects. It receives neither
human sentiment/presence labels nor reference ID sets.

## Data

The fixed benchmark has 100 questions and up to ten saved `hybrid_rerank` passages
per question. Labels derive from 1,000 human-reviewed passages. These are
condensed annotated rankings, not production retrieval top-k results.
See [the annotation documentation](data/generation_annotations/README.md).
Optional uploaded subsets must preserve the original questions and evidence.

## Command line

```bash
# Set GENERATION_API_KEY privately, then run the same new experiment:
python run_fact_experiments.py --output experiments/fact_smoke --limit 3

# Recompute ID metrics without any model calls:
python evaluate_fact_selection.py predictions.jsonl
```

See [EXPERIMENTS.md](EXPERIMENTS.md), Section 13, for the current webpage protocol.
Earlier sections and `run_generation_experiments.py` / `rejudge_results.py`
preserve the previous LLM-judge experiment for reproducibility.

```bash
python -m unittest discover -s tests -v
```

Tests use scripted clients and do not establish real model performance.
