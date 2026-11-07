# AI6127: Evidence-grounded Review Summarization

Final experiment code for comparing four RAG summarization methods with fixed retrieved evidence and deterministic fact-ID evaluation against references refined through human review.

Requires **Python 3.10+**. The project uses only the Python standard library; no evaluator model or third-party Python packages are required.

Unless stated otherwise, run commands from the directory containing this README. In the group repository, that directory is `code/generation/rag_summary/`. The integration examples below explicitly run from `code/`.

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

### Latest Review Updates

Live generation defaults to `context_policy="latest_update"`. Before context truncation, it selects the newest available snapshot of each review using `metadata.updated_at_utc` (or `timestamp_updated`), falling back to the creation timestamp. Different reviewers remain separate evidence sources.

When a review contains explicit substantive `Edit:` or `Update:` blocks, only its latest visible update and continuation are provided to the model. Dated blocks use their dates when all updates are dated; otherwise their order determines the latest update. Repeated and numbered headings such as `Edit Edit:` and `(Edit 2:` are supported. Typo/phrasing-only notes do not replace the substantive review. If timestamps show an edit but no update boundaries are identifiable, the latest full snapshot is retained.

For example, the Solar Crown review's final February 2026 update is provided without its earlier RTX 3070/60 FPS description. Conditions remaining in the selected update are preserved. Original stored reviews and citation IDs are unchanged. The same filtered evidence is used for generation, quote validation, verification, and revision. Available creation/update timestamps accompany the evidence; outputs record `context_policy` and the actual `context_evidence` for inspection. This uses the latest review content available in the retrieved hits.

To explicitly use full reviews, pass `context_policy="full_review"` or the CLI option `--context-policy full_review`. The fixed 100-question experiment, including the experiment web interface, uses `full_review` to match its existing reference annotations and reproduce the reported results. Evaluating the new policy requires reference summaries and fact labels for the filtered evidence.

### Helpfulness and Publication Age

Live latest-update generation also provides `helpful_votes`, `review_age_days`, and `helpful_votes_per_day` when the source fields are available. The weight is:

```text
helpful_votes_per_day = helpful_votes / max(days_since_first_publication, 1)
```

The one-day floor covers newly published reviews. The denominator uses the creation timestamp, even for edited reviews, because votes belong to the whole review. Missing/invalid counts or missing/future publication timestamps do not produce a weight. Raw Steam `votes_up` and `timestamp_created` are accepted alongside the normalized fields. Review text, citation IDs, and retrieval order are preserved.

The prompt uses these signals to help prioritize relevant supported information. `is_high_helpfulness` identifies reviews with at least ten helpful votes and at or above the nearest-rank upper quartile of distinct reviews in the visible context. Only these reviews may be called “highly upvoted” when useful to the answer. Lower-vote evidence is integrated without vote attribution; exact vote counts and rates stay out of answer prose. Votes do not count as additional reviewers or establish agreement with each assertion.

Generation now aims for one coherent answer: lead with the response to the question, merge overlapping evidence, and connect benefits, limitations and disagreements naturally. The JSON `claims` array contains its sentences in reading order, with citations and quotes preserved. The prompts prohibit adding unsupported history, present popularity, recent improvements or new mod development merely to make the answer sound fluent.

All stages share one UTC reference time, recorded as `context_as_of_utc` together with the formula and selected context. For reproducible inputs, pass `as_of="2026-11-06T12:00:00Z"` to `generate_summary` or `build_context`, or use the CLI option `--as-of`. The fixed `full_review` benchmark retains its original evidence payload and annotations. Current prompts use the synthesis instructions; archived generation results and their recorded prompt hashes remain available for reproduction.

## Connect to the Group Retriever

The group repository documents the handoff in [code/generation/README.md](https://github.com/GSrihariS/RAG_ASPA_Game_Retrieval/blob/main/code/generation/README.md). This module lives at `code/generation/rag_summary/`, alongside the separate aspect-based sentiment analysis module. The summarizer's callable is `generate_summary(query, hits)`; no `generation.answer` wrapper is required. Sentiment analysis can run separately on the same hits.

Run the following integration examples from the group repository's **`code/` directory**. They use the existing retriever dependencies; the summarizer itself adds no Python packages.

### Input and Output Contract

Pass the original user question and the final ranked `hits` list from `app.py` directly to the summarizer. Use the same list displayed as the top review results.

The ready-to-use app in [`integration/app.py`](integration/app.py) requires an explicit game selection before searching. Both sparse and dense retrieval receive `filters={"game_name": selected_games}`. Candidate and final-evidence checks exclude other games, and `render_summary(query, hits, games)` checks the scope again before generation. Select multiple games for comparisons. Each app hit needs `metadata.game_name` to pass these checks. The saved answer includes the applied `retrieval_filters`.

| Retriever field | Use in summarization |
|---|---|
| Original `query` | The question to answer; use this rather than the expanded retrieval query |
| `hits[*].chunk_id` | Required; retained as the original citation ID |
| `hits[*].text` | Required; the review evidence used to generate claims |
| `hits[*].metadata.game_name` | Optional; identifies the game discussed by the passage |
| `hits[*].metadata.source_url` | Optional; retained as the review link in the output citations |
| Review identity, timestamps, and chunk offsets in `metadata` | Used to select the latest snapshot/update; available timestamps accompany model evidence |
| `hits[*].metadata.helpful_votes` or `votes_up` | Recorded helpful vote count; used with publication age as an auxiliary weight in live generation |
| List order | Preserved when building the context; duplicate passage IDs are removed |
| Scores, ranks, `retrieved_by`, and other metadata | Not included in the model context |

Review identity uses Steam chunk IDs, or `metadata.review_id` with `appid`/`game_name`, or `document_id`. Temporal selection reads `created_at_utc`/`updated_at_utc` (or `timestamp_created`/`timestamp_updated`), `chunk_index`, and `char_start` when available. `metadata.source` and `metadata.title` can remain in each hit. If a source URL is absent, citations still retain the original chunk ID.

The return value includes:

- `summary`: the answer with inline `[chunk_id]` citations.
- `claims`: the retained claim text, supporting `chunk_ids`, and evidence quotes for B/C/D.
- `citations`: unique cited `chunk_id` values and their available `source_url` values.
- `status`: `ok`, `abstained`, or `no_evidence`.
- `usage`, `latency_seconds`, `stage_calls`, and `claim_audit`: generation usage and verification records.
- `context_policy` and, for live latest-update generation, `context_evidence`: the evidence policy and selected model input.
- `context_as_of_utc` and `review_weight_formula`: the reference time and formula for live helpfulness weights.

For live retrieval, omit `fact_candidates`. Original chunk IDs provide citations for any retrieved review. The fixed-benchmark runner supplies its own fact catalog when running the structured evaluation.

### API Configuration

Set these environment variables in the shell or hosting environment before starting the group app:

| Variable | Value |
|---|---|
| `GENERATION_API_KEY` | Your generation provider's API key |
| `GENERATION_BASE_URL` | For example, `https://api.deepseek.com` |
| `GENERATION_MODEL` | For example, `deepseek-v4-pro` |

Keep the key local. It is read from the environment and is not part of the retriever output or generated result files.

### Add Summarization to the Streamlit App

The tested replacement app and patch are provided in [`integration/`](integration/README.md). For a manual integration, require `games` before invoking search and pass the game filter to both retrievers; the existing `search()` accepts the selected games. The block below adds the summary within the **"Sentiment analysis / generated answer"** container. Keep the separate sentiment-analysis module's integration alongside it. The app already defines `query`, `hits`, `games`, `json`, and `st`.

```python
import os

from paths import GENERATION_RESULTS_DIR
from generation.rag_summary.rag_generator import (
    GenerationError,
    OpenAICompatibleClient,
    generate_summary,
    safe_error,
)

st.markdown("**Evidence-grounded review summary**")
if not games:
    st.info("Select at least one target game before generating a summary.")
    st.stop()
hits = [h for h in hits if (h.get("metadata") or {}).get("game_name") in games]
if not hits:
    st.warning("No reviews match the selected games.")
    st.stop()
evidence_key = (query, tuple(sorted(games)), tuple(
    (h["chunk_id"], h["text"], json.dumps(h.get("metadata") or {}, sort_keys=True))
    for h in hits
))

if st.button("Generate summary", key="rag_summary_generate"):
    try:
        with st.spinner("Generating an answer from the retrieved reviews..."):
            result = generate_summary(
                query,
                hits,
                approach="improved",
                model=os.getenv("GENERATION_MODEL", "deepseek-v4-pro"),
                client=OpenAICompatibleClient(
                    base_url=os.getenv("GENERATION_BASE_URL", "https://api.deepseek.com")
                ),
                max_claims=4,
            )
        result["retrieval_filters"] = {"game_name": list(games)}
        st.session_state["rag_summary_answer"] = (evidence_key, result)
        output_dir = GENERATION_RESULTS_DIR / "rag_summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "latest_answer.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except GenerationError as error:
        st.error(safe_error(error)["error_message"])

saved = st.session_state.get("rag_summary_answer")
if saved is not None and saved[0] == evidence_key:
    result = saved[1]
    st.markdown(result["summary"])
    for citation in result["citations"]:
        if citation["source_url"]:
            st.link_button(citation["chunk_id"], citation["source_url"])
    st.download_button(
        "Download generated answer",
        json.dumps(result, ensure_ascii=False, indent=2),
        file_name="rag_summary.json",
        mime="application/json",
        key="rag_summary_download",
    )
```

Start the existing group interface from `code/`:

```bash
streamlit run app.py
```

Select the target game or games, search, then click **Generate summary**. The button controls model calls; Streamlit reruns reuse the stored answer for the same question, game scope and evidence. Results are saved through `paths.py` to `data/generation_results/rag_summary/latest_answer.json`.

### Use a Downloaded Retrieval Export

The group interface's **Download results (JSON)** button produces a `search_results.json` object containing `query` and `results`. The generator accepts this export directly; `mode`, `filters`, `params`, `expansion`, and `timings_ms` require no conversion.

Save the export as `data/retrieval_results/search_results.json`. After setting the environment variables above, run from `code/`:

```bash
python -c "from paths import GENERATION_RESULTS_DIR; (GENERATION_RESULTS_DIR / 'rag_summary').mkdir(parents=True, exist_ok=True)"
python generation/rag_summary/rag_generator.py ../data/retrieval_results/search_results.json --approach improved --max-claims 4 --output ../data/generation_results/rag_summary/answer.json
```

For programmatic loading, use:

```python
from generation.rag_summary.rag_generator import load_retrieval
from paths import RETRIEVAL_RESULTS_DIR

query, hits = load_retrieval(RETRIEVAL_RESULTS_DIR / "search_results.json")
```

### Retrieve Without the UI

Use the group's fusion and reranking methods to obtain the ranked list:

```python
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_expansion import expand_query
from retrieval.reranker import Reranker

query = "How is the soundtrack in Hollow Knight?"
expansion = expand_query(query, mode="hierarchical")
fused = HybridRetriever().retrieve_fused(
    query,
    top_k=30,
    candidates=20,
    filters={"game_name": ["Hollow Knight"]},
    sparse_query=expansion.sparse,
    dense_query=expansion.dense,
)
hits = Reranker().rerank(query, fused, top_k=5)
# Pass query and hits to generate_summary as in the Streamlit example.
```

`HybridRetriever.retrieve()` returns separate sparse and dense lists. `retrieve_fused()` and `Reranker.rerank()` produce the ranked list consumed by the summarizer.

### Live Answers and Benchmark Evaluation

The integration above generates answers from the current retrieval results. The experiment web interface launched by `launch_web.py` runs the fixed 100-question benchmark with its saved evidence and reference sets. Its upload option accepts unchanged benchmark subsets. New questions or different retrieved passages require corresponding references before fact-selection metrics can be computed.

The module's bundled benchmark data and final experiment results remain inside `rag_summary/` for reproduction. Live group-app outputs use the shared `GENERATION_RESULTS_DIR` path shown above.

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

- `data/reference_summaries/reviewed/approved_references.jsonl`: 100 human-reviewed English summary references, including two translated Chinese edits. These answer-level references are ready for ROUGE evaluation and are separate from the fact-ID labels below.
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
review_context.py           Latest review updates and helpfulness/age signals
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
