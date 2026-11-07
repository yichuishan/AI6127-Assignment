# AI6127: Evidence-grounded Review Summarization

Experiment code for a 2 × 2 study of few-shot prompting and extractive context compression, evaluated with ROUGE-1/2/L, token usage, latency and cost. The 100 reviewed questions are split into 20 development questions and 80 test questions.

Requires **Python 3.10+**. Generation and ROUGE scoring use only the Python standard library; no evaluator model or third-party runtime packages are required.

Unless stated otherwise, run commands from the directory containing this README. In the group repository, that directory is `code/generation/rag_summary/`. The integration examples below explicitly run from `code/`.

## Run the Web Interface

```bash
python launch_web.py
```

Once the browser opens, enter the generation API URL, model name, and API key. Set the number of questions, repetitions, and maximum claims per answer, then start the experiment to view progress and download the results ZIP.

The default settings are `https://api.deepseek.com` and `deepseek-v4-pro`, with thinking automatically disabled for the official DeepSeek V4 API. Start with the default three questions. For the final experiment, select the test split, 80 questions, one repetition, and at most four claims per answer. The development split contains 18 runnable questions; two additional development questions are reserved as demonstrations. Restart any old server after updating and refresh the page.

API keys remain in process memory and are excluded from result files and browser storage. Keep the terminal open while using the interface; press Ctrl+C to stop the server.

## Generation Methods

| Method | Few-shot demonstrations | Current evidence |
|---|---|---|
| Baseline | None | Complete fixed reviews |
| Few-shot | Two fixed development examples | Complete fixed reviews |
| Compression | None | Locally selected review sentences |
| Few-shot + compression | Two fixed development examples | Locally selected review sentences |

All four methods share the baseline prompt, source-citation schema, model, metadata reference time and output limit. Each question/method/repetition makes one independent generation call. There is no draft sharing, self-check, revision or evaluator call in this experiment.

The examples teach coherent synthesis, preserving qualifications and contrasting experiences. They use development questions q06 and q24 and their reviewed English answers. The current question's reference answer is never sent to generation. Example IDs cannot be scored in either split.

Context compression fits TF-IDF locally to the current review sentences and ranks them against the question. A greedy relevance/redundancy criterion uses weight 0.8 for relevance and 0.2 for similarity to selected sentences. It first attempts to retain each sentence's immediate neighbors, then the complete sentence if the window exceeds the budget. Source order, IDs, metadata and exact text are retained. Separate extracts are joined with a visible gap marker. The default budget is 60% of original review-text characters; one indivisible sentence may exceed that budget and is flagged. A ratio of 1 retains the original context exactly. No reference text, aspect label or model is used for compression.

Actual API input tokens include prompts, metadata and demonstrations. The text-retention ratio concerns review text only. Both are reported independently.

The generation interface is provided by `rag_generator.py`:

```python
from rag_generator import generate_summary

result = generate_summary(
    question,
    ranked_passages,
    model="deepseek-v4-pro",
    max_claims=4,
)
```

Each entry in `ranked_passages` requires `chunk_id` and `text`, with optional `game_name` and `source_url` fields inside `metadata`. The ROUGE runner sends the question, review evidence and source metadata. Few-shot configurations also send the two development demonstrations. The current question's reference answer and fact candidates are never included.

### Latest Review Updates

Live generation defaults to `context_policy="latest_update"`. Before context truncation, it selects the newest available snapshot of each review using `metadata.updated_at_utc` (or `timestamp_updated`), falling back to the creation timestamp. Different reviewers remain separate evidence sources.

When a review contains explicit substantive `Edit:` or `Update:` blocks, only its latest visible update and continuation are provided to the model. Dated blocks use their dates when all updates are dated; otherwise their order determines the latest update. Repeated and numbered headings such as `Edit Edit:` and `(Edit 2:` are supported. Typo/phrasing-only notes do not replace the substantive review. If timestamps show an edit but no update boundaries are identifiable, the latest full snapshot is retained.

For example, the Solar Crown review's final February 2026 update is provided without its earlier RTX 3070/60 FPS description. Conditions remaining in the selected update are preserved. Original stored reviews and citation IDs are unchanged. The selected evidence is used for generation and source-citation validation. Available creation/update timestamps accompany the evidence; outputs record `context_policy` and the actual `context_evidence` for inspection. This uses the latest review content available in the retrieved hits.

To explicitly use full reviews, pass `context_policy="full_review"` or the CLI option `--context-policy full_review`. The ROUGE experiment first uses `full_review` with `include_review_metadata=True` to construct the reference-matched original context. Compression methods then extract sentences from that context. Latest-update filtering is separate from this study.

### Helpfulness and Publication Age

Live latest-update generation also provides `helpful_votes`, `review_age_days`, and `helpful_votes_per_day` when the source fields are available. The weight is:

```text
helpful_votes_per_day = helpful_votes / max(days_since_first_publication, 1)
```

The one-day floor covers newly published reviews. The denominator uses the creation timestamp, even for edited reviews, because votes belong to the whole review. Missing/invalid counts or missing/future publication timestamps do not produce a weight. Raw Steam `votes_up` and `timestamp_created` are accepted alongside the normalized fields. Review text, citation IDs, and retrieval order are preserved.

The prompt uses these signals to help prioritize relevant supported information. `is_high_helpfulness` identifies reviews with at least ten helpful votes and at or above the nearest-rank upper quartile of distinct reviews in the visible context. Only these reviews may be called “highly upvoted” when useful to the answer. Lower-vote evidence is integrated without vote attribution; exact vote counts and rates stay out of answer prose. Votes do not count as additional reviewers or establish agreement with each assertion.

Generation now aims for one coherent answer: lead with the response to the question, merge overlapping evidence, and connect benefits, limitations and disagreements naturally. The JSON `claims` array contains its sentences in reading order, with citations and quotes preserved. The prompts prohibit adding unsupported history, present popularity, recent improvements or new mod development merely to make the answer sound fluent.

All stages share one UTC reference time, recorded as `context_as_of_utc` together with the formula and selected context. For reproducible inputs, pass `as_of="2026-11-06T12:00:00Z"` to `generate_summary` or `build_context`, or use the CLI option `--as-of`. The benchmark retains the original evidence and saves the actual compressed context separately. Current prompts use the synthesis instructions; archived generation results and their recorded prompt hashes remain available for reproduction.

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
- `claims`: the answer sentences and supporting `chunk_ids`.
- `citations`: unique cited `chunk_id` values and their available `source_url` values.
- `status`: `ok`, `abstained`, or `no_evidence`.
- `usage`, `latency_seconds`, `stage_calls`, and `claim_audit`: generation usage and citation-validation records.
- `context_policy` and `context_evidence`: the evidence policy and selected model input.
- `context_as_of_utc` and `review_weight_formula`: the reference time and formula for live helpfulness weights.

Original chunk IDs remain available for source citations and are removed from the answer text before ROUGE scoring. Generation uses one model call, with optional `few_shot_examples` and `compression_ratio` arguments. The returned `approach` is inferred from those options.

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
python generation/rag_summary/rag_generator.py ../data/retrieval_results/search_results.json --max-claims 4 --output ../data/generation_results/rag_summary/answer.json
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

The integration above generates answers from the current retrieval results. The experiment web interface launched by `launch_web.py` uses a frozen 20/80 development/test split of the fixed 100-question benchmark. Its upload option accepts unchanged subsets belonging to the selected partition, excluding the two demonstration questions. New questions or different retrieved passages require matching references before ROUGE comparison.

The module's bundled benchmark data and archived experiment results remain inside `rag_summary/` for reference. Live group-app outputs use the shared `GENERATION_RESULTS_DIR` path shown above.

## Command-line ROUGE Experiments

Set `GENERATION_API_KEY` or `OPENAI_API_KEY` in your own terminal, then run:

```bash
python run_generation_experiments.py --output experiments/runs/fewshot_compression_test --split test --limit 80 --compression-ratio 0.6 --repeats 1 --max-claims 4 --base-url https://api.deepseek.com --model deepseek-v4-pro
```

For a trial run, use `--limit 3`. For development, use `--split development --limit 18`. Use a new, empty output directory. API keys are read from the local environment and excluded from outputs.

The allowed experiment methods are `baseline`, `few_shot`, `compressed`, and `combined`. Self-check methods are not accepted by the current runner or web API.

Outputs include:

- `generation_results.json` and `checkpoint.jsonl`: each answer, source citations, its reference answer, all nine ROUGE precision/recall/F1 scores, and execution metrics.
- `summary.csv`: macro means and valid counts for ROUGE-1/2/L, latency, model calls, tokens and optional cost estimates.
- `report.md` and `report_data.json`: summary tables, five paired method comparisons with query-bootstrap confidence intervals, and the factorial interaction (combined − few-shot − compression + baseline). ROUGE, input tokens, latency and optional costs receive paired comparisons.
- `evaluation_references.jsonl` and `input.jsonl`: the current scoring references and original cases.
- `split.json` and `few_shot_examples.json`: the frozen split and exact demonstrations. Each result also records actual evidence; compressed results retain full original evidence and extract character spans.
- `manifest.json`: the protocol, split, method settings, compression ratio, scoring options, metadata reference time, and data/prompt/code hashes.

The web interface runs this same workflow and includes these files in its results ZIP.

## ROUGE Scoring

The scorer uses the 100 English answers in `data/reference_summaries/reviewed/approved_references.jsonl`. Each must be marked human-reviewed and match the question and evidence revision.

| Metric | Comparison |
|---|---|
| ROUGE-1 | Overlap of word-token counts |
| ROUGE-2 | Overlap of consecutive token-pair counts |
| ROUGE-L | Longest common subsequence of the complete answer and reference |

Source-ID citation markers are removed before scoring. Tokenization lowercases text and keeps ASCII English letter/digit tokens, without stemming. This matches the default tokenizer of Google's `rouge-score` with `use_stemmer=False`. ROUGE-L is whole-answer LCS, not ROUGE-Lsum. The dependency-free implementation has been cross-checked against `rouge-score==0.1.2` on 506 text pairs, covering all nine scores.

Scores in JSON and CSV are between 0 and 1. The web page and Markdown report show F1 as percentages. Aggregation takes the mean of completed-answer scores; failed requests are recorded separately. Each completed answer, including an abstention, is scored from its actual text. Repetitions remain grouped by question in bootstrap comparisons.

The original benchmark evidence and scoring references remain fixed. Compression selects sentences from that original context, without changing the reference target. Few-shot methods use only the two development demonstrations. No current-question reference answer, fact candidate or sentiment label is sent to generation. Uploaded cases must preserve their original question, passages, order and metadata and belong to the selected split.

## Recompute ROUGE Without API Calls

Score a downloaded experiment ZIP, results directory, or saved result JSON:

```bash
python evaluate_generation.py generation-experiment.zip --output experiments/runs/rouge_rescored
```

Current full and compressed results can be rescored without regeneration. The scorer validates the experiment protocol, actual evidence, compression spans and frozen demonstration IDs. Original result files are never overwritten.

## Data and Historical Fact-ID Evaluation

- `data/reference_summaries/reviewed/approved_references.jsonl`: 100 approved English reference answers for current ROUGE scoring.
- `experiments/steam_generation_eval.jsonl`: 100 fixed questions, each with ten saved annotated review passages from `hybrid_rerank` rankings.
- `data/generation_annotations/`: the original 1,000 annotated passages, 2,898 review–aspect facts and historical fact-selection targets.
- `data/generation_annotations/human_review.json`: 40 query-specific decisions used in the earlier fact-selection study.

Historical Precision/Recall/F1, coverage-unit recall and polarity-contrast scores remain in the saved report artifacts. The old runner, scorer, compatibility commands and self-check prompts have been removed. Their source is available in Git history at commit `7f4537d`.

## Archived Fact-selection Results

`experiments/final/` preserves the earlier generation outputs and fact-selection scores used in the report. The values below are historical fact-ID scores, not ROUGE:

| Method | Precision | Recall | F1 | Unit recall | Contrast coverage |
|---|---:|---:|---:|---:|---:|
| A | 85.40 | 72.71 | 75.82 | 88.73 | 56.25 |
| B | 88.03 | 69.68 | 75.12 | 87.18 | 50.00 |
| C | 89.49 | 68.67 | 74.74 | 87.05 | 50.00 |
| D | 89.20 | 69.58 | 75.23 | 87.05 | 50.00 |

A/B each completed 100 tasks; C/D each completed 99. See the saved outputs and [final results CSV](experiments/final/summary.csv) for individual results, valid counts, token usage, and timing.

## Repository Structure

```text
rag_generator.py            Single-call generation, citations, optional examples/compression
review_context.py           Latest review updates and helpfulness/age signals
prompts/                    Baseline and few-shot instructions
context_compression.py      Local TF-IDF sentence selection and traceable spans
generation_experiment_design.py Frozen split, demonstrations and method settings
run_generation_experiments.py Few-shot × compression ROUGE experiment runner
evaluate_generation.py      Deterministic ROUGE scorer and offline result evaluation
experiment_web.py            Web backend, progress tracking, and ZIP export
launch_web.py                Local launch entry point
web/                        Web frontend
data/reference_summaries/   Drafts and finalized human-reviewed English answers
data/generation_experiment/ Fixed 20/80 split and two development demonstrations
data/generation_annotations/ Original annotations and historical scoring references
experiments/                Fixed inputs, source audit, and final results
tests/                      Generation, scoring, web, and data-integrity tests
```

## Validation

```bash
python -m unittest discover -s tests -v
```

Tests use scripted models and saved outputs; they do not call paid APIs. Optional frontend checks:

```bash
node tests/experiment_web_ui.cjs
node tests/reference_review_ui.cjs
```

An optional real-browser smoke test uses Playwright and an available Chromium binary (`CHROMIUM_PATH` can override the binary). It runs a temporary localhost server and a scripted model:

```bash
python tests/browser_experiment.py
```

`tests/validate_rouge_against_reference.py` reproduces the optional cross-check when `rouge-score==0.1.2` is installed in a separate test environment. This package is not needed to run generation or scoring.
