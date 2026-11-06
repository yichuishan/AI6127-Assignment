# AI6127: Evidence-grounded Review Summarization

Final experiment code for comparing four RAG summarization methods with fixed retrieved evidence and deterministic fact-ID evaluation against references refined through human review.

Requires **Python 3.10+**. The project uses only the Python standard library; no evaluator model or third-party Python packages are required.

## Run the Web Interface

```bash
python launch_web.py
```

Once the browser opens, enter the generation API URL, model name, and API key. Set the number of questions, repetitions, and maximum claims per answer, then start the experiment to view progress and download the results ZIP.

The default settings are `https://api.deepseek.com` and `deepseek-v4-pro`, with thinking automatically disabled for the official DeepSeek V4 API. Start with the default three questions. For the full experiment, select 100 questions, one repetition, and at most four claims per answer.

API keys remain in process memory and are excluded from result files and browser storage. Keep the terminal open while using the interface; press Ctrl+C to stop the server.

## Generation Methods

| Method | Workflow |
|---|---|
| A · Baseline | Generate claims with source citations and fact IDs |
| B · Evidence quotes | Strengthen the evidence instructions and require validated verbatim quotes |
| C · Verify-delete | Start from B's draft, check each claim, and delete claims that fail verification |
| D · Verify-revise | Start from B's draft, repair claims that fail verification, and check the revisions again |

B/C/D share the initial draft. C/D use the generation model for self-checking; final scoring uses the same deterministic algorithm for every method, with no separate evaluator model.

The generation interface is provided by `rag_generator.py`:

```python
from rag_generator import generate_summary

result = generate_summary(
    question,
    ranked_passages,
    approach="improved",
    model="deepseek-v4-pro",
    max_claims=4,
)
```

Each entry in `ranked_passages` requires `chunk_id` and `text`, with optional `game_name` and `source_url` fields inside `metadata`. For structured experiments, the runner adds eight aspect candidates per review without presence or sentiment labels. Reference answers and human sentiment labels are excluded from model inputs.

## Command-line Experiments

Set `GENERATION_API_KEY` in your own terminal, then run:

```bash
python run_fact_experiments.py --output experiments/runs/full --limit 100 --repeats 1 --max-claims 4 --base-url https://api.deepseek.com --model deepseek-v4-pro
```

For a trial run, replace `--limit 100` with `--limit 3`. Use a new, empty output directory.

Outputs include complete generation results, per-task checkpoints, predicted IDs, the reference sets used, CSV metrics, summaries, and configuration hashes.

## Recompute Scores Without API Calls

Rescore a downloaded experiment ZIP:

```bash
python recompute_results.py generation-experiment.zip --output experiments/runs/rescored
```

Reproduce the report results without an API key:

```bash
python recompute_results.py experiments/final/generation_results.json --output experiments/runs/report_reproduction
```

Score a prediction JSONL file directly:

```bash
python evaluate_fact_selection.py experiments/final/predictions.jsonl --output experiments/runs/prediction_metrics.json
```

Each prediction line contains `id` and `selected_fact_ids`, with optional `approach`, `repeat`, and `summary` fields. The scripts use the final reference sets included in this repository.

## Data and Scoring

- `experiments/steam_generation_eval.jsonl`: 100 fixed questions, each with up to ten annotated review passages from saved `hybrid_rerank` rankings.
- `data/generation_annotations/`: 1,000 annotated source passages, 2,898 review–aspect facts, and the final reference sets.
- `data/generation_annotations/human_review.json`: 40 user-adjudicated decisions across 16 questions, comprising 22 core facts, seven optional background facts, and 11 excluded facts.

Core facts belong to the required reference set. Optional background facts are neither required for coverage nor penalized as incorrect selections. Excluded facts count as incorrect when selected. Items outside this review retain the original annotation-matching rules.

Metrics include fact-selection Precision/Recall/F1, coverage-unit recall, polarity-contrast coverage, execution time, model calls, and token usage. Coverage units group facts by game, aspect, and sentiment. The final references contain 591 required query–fact pairs, 238 coverage units, and 27 positive/negative contrast groups across 24 questions. Condition retention is not scored because the corresponding human annotations are unavailable.

These metrics evaluate structured fact selection. See the [data documentation](data/generation_annotations/README.md) for sources, fields, and scoring rules. The web interface accepts subsets of the original benchmark with unchanged questions and evidence.

## Final Experiment Results

`experiments/final/` contains the actual generation outputs and revised scores used in the report. Scores below are percentages macro-averaged across applicable questions:

| Method | Precision | Recall | F1 | Unit recall | Contrast coverage |
|---|---:|---:|---:|---:|---:|
| A | 85.40 | 72.71 | 75.82 | 88.73 | 56.25 |
| B | 88.03 | 69.68 | 75.12 | 87.18 | 50.00 |
| C | 89.49 | 68.67 | 74.74 | 87.05 | 50.00 |
| D | 89.20 | 69.58 | 75.23 | 87.05 | 50.00 |

A/B each completed 100 tasks; C/D each completed 99. See the saved outputs and [final results CSV](experiments/final/summary.csv) for individual results, valid counts, token usage, and timing.

## Repository Structure

```text
rag_generator.py            Generation, citation validation, self-checking, and revision
prompts/                    A/B, fact-selection, verification, and revision prompts
run_fact_experiments.py      Fixed-benchmark experiment runner
experiment_web.py            Web backend, progress tracking, and ZIP export
launch_web.py                Local launch entry point
web/                        Web frontend
evaluate_fact_selection.py  Shared deterministic scorer
recompute_results.py        Offline rescoring of saved outputs
data/generation_annotations/ Final data and human review records
experiments/                Fixed inputs, source audit, and final results
tests/                      Generation, scoring, web, and reproduction tests
```

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests use scripted models and saved final outputs; they do not call paid APIs.
