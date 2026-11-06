# Downstream task and generation

This directory contains the downstream modules that consume the ranked Steam review passages returned by the retriever.

| Component | Location | Purpose |
|---|---|---|
| Aspect-based Sentiment Analysis | Provided by the classification contributor | Predict aspect-level sentiment from the review evidence |
| Evidence-grounded RAG summarization | `generation/rag_summary/` | Answer the user question with review citations |

The two components can use the same retrieved passages. The summarizer reads the review text directly and does not require predicted sentiment labels from the classification component. Its implementation and experiment instructions are documented in `rag_summary/README.md`.

## Connecting to the UI

`code/app.py` already computes `query` and a final ranked `hits` list. Its placeholder titled **"Sentiment analysis / generated answer"** is the integration point. The summarization function is:

```python
from generation.rag_summary.rag_generator import generate_summary

result = generate_summary(query, hits, approach="improved", max_claims=4)
```

Configure the API environment variables below before making this call. Run integration commands from **`code/`**, so imports from `generation`, `retrieval`, and `paths` resolve.

### Retrieval input

Pass the original question and the same final `hits` list displayed by the app. Each hit is a dictionary with:

| Key | Content and use |
|---|---|
| `chunk_id` | Required. Original passage ID, retained in citations |
| `text` | Required. Review evidence used to generate the answer |
| `metadata` | Optional dictionary containing source information |
| `metadata.game_name` | Game identity used when interpreting the passage |
| `metadata.source_url` | Original review URL retained in the output citations |
| `metadata.created_at_utc`, `metadata.updated_at_utc` | Review timestamps; Unix `timestamp_created`/`timestamp_updated` are also accepted |
| `metadata.review_id`, `appid`, `document_id`, `chunk_index`, `char_start` | Optional identity and chunk-position information for temporal selection |

Additional metadata such as `source` and `title` can remain in the hit. Rank and score fields such as `retrieved_by`, `sparse_rank`, `dense_rank`, `rrf_rank`, `rrf_score`, and `score` are not added to the model context. The ranked list order is preserved, duplicate passage IDs are removed, and the original user question is used rather than the expanded retrieval query.

Live generation defaults to `context_policy="latest_update"`: among multiple snapshots of one review, only the newest available snapshot is retained. Explicit substantive `Edit:`/`Update:` sections are reduced to the latest visible update and continuation, including updates spanning multiple supplied chunks. Earlier opinions are excluded from generation and quote validation. Wording-only edit notes do not replace the review. If update boundaries are absent, the latest full snapshot is used. Available review timestamps and an `evidence_scope` field accompany the text. Original stored reviews and citation IDs are preserved. Update the complete `rag_summary/` module, including `review_context.py`, when installing this behavior.

No conversion is required between the app's final `hits` list and the summarizer. BM25-only, dense-only, fused, and reranked results use the same input contract.

### API configuration

Set the following environment variables in the shell or hosting environment before starting the app:

| Variable | Example or purpose |
|---|---|
| `GENERATION_API_KEY` | Your generation provider's API key |
| `GENERATION_BASE_URL` | `https://api.deepseek.com` |
| `GENERATION_MODEL` | `deepseek-v4-pro` |

The API must support OpenAI-compatible Chat Completions and JSON object output. The summarizer reads credentials from the environment. Keys are not included in generated result files. Official DeepSeek V4 thinking is disabled automatically by the default client configuration.

### Streamlit integration

Replace the contents of the existing placeholder container with the block below. Keep the classification component's UI alongside the summary. The app already imports `json` and `streamlit as st`, and defines `query` and `hits`.

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
evidence_key = (query, tuple(
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

Run the existing search interface from `code/`:

```bash
streamlit run app.py
```

Search for reviews, then click **Generate summary**. Only this button initiates generation. Streamlit reruns reuse the saved answer when the question and evidence are unchanged. The example saves the latest answer to `data/generation_results/rag_summary/latest_answer.json` through the shared `paths.py` configuration.

### Generation output

The returned dictionary contains:

| Field | Content |
|---|---|
| `summary` | Answer text with inline `[chunk_id]` citations |
| `claims` | Retained claims, supporting `chunk_ids`, and evidence quotes for B/C/D |
| `citations` | Unique cited passage IDs and available source URLs |
| `status` | `ok`, `abstained`, or `no_evidence` |
| `usage` | Available input/output token counts |
| `latency_seconds` | Generation workflow time |
| `stage_calls`, `claim_audit` | Generation and self-verification records |
| `context_policy`, `context_evidence` | The selected policy and actual latest-update context used by the model |

The `improved` configuration requires verbatim evidence quotes and validates them against the supplied passages. The other available configurations are `baseline`, `verify_delete`, and `verify_revise`. The latter two use the generation model for internal self-checking. Live generation uses the original chunk IDs for citations and does not require the benchmark's `fact_candidates`.

## Using retrieval directly

Run from `code/`. The following example follows the group's query expansion, fusion, and reranking pipeline, then generates an answer from the final results:

```python
from generation.rag_summary.rag_generator import generate_summary
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
result = generate_summary(query, hits, approach="improved", max_claims=4)
```

`HybridRetriever.retrieve()` returns separate sparse and dense lists. Use the final list from fusion or reranking as the generation input.

### Using the UI's JSON export

The search interface's **Download results (JSON)** button produces `search_results.json` containing `query` and `results`. The generator accepts this structure directly. Save the export to `data/retrieval_results/search_results.json`, set the API environment variables, and run from `code/`:

```bash
python -c "from paths import GENERATION_RESULTS_DIR; (GENERATION_RESULTS_DIR / 'rag_summary').mkdir(parents=True, exist_ok=True)"
python generation/rag_summary/rag_generator.py ../data/retrieval_results/search_results.json --approach improved --max-claims 4 --output ../data/generation_results/rag_summary/answer.json
```

Fields such as `mode`, `filters`, `params`, `expansion`, and `timings_ms` remain in the retrieval export and are not passed to the model as review evidence.

## Evaluation

The live integration above answers questions using the current retrieval results. The summarization experiment uses a separate fixed 100-question benchmark with saved contexts and human-reviewed reference sets. It measures fact-selection Precision/Recall/F1, coverage-unit recall, and polarity-contrast coverage using deterministic code. Final scoring does not call an evaluator model.

The experiment runner and its web interface explicitly use `context_policy="full_review"` to match the existing references. Live summarization uses `latest_update` by default. Reference annotations must match the filtered context before evaluating the new policy.

To reproduce the saved summarization results without API calls, run from `code/`:

```bash
python generation/rag_summary/recompute_results.py generation/rag_summary/experiments/final/generation_results.json --output ../data/generation_results/rag_summary/report_reproduction
```

Use a new output directory. Full experiment settings, benchmark data, results, and additional commands are described in `rag_summary/README.md`. New questions or changed retrieval contexts need corresponding reference annotations before benchmark selection metrics can be computed.

The classification contributor should document the classification interface, execution commands, and evaluation alongside this summarization interface.

## Conventions

- Run group integration commands from `code/` and take shared data locations from `paths.py`.
- Write integrated generation outputs and task-specific evaluation results under `GENERATION_RESULTS_DIR`, using separate subdirectories for each component. The summarizer uses `data/generation_results/rag_summary/`.
- The summarizer's bundled benchmark data and archived results remain inside `rag_summary/` for reproduction; the integration examples write new outputs to the shared data directory.
- Add any new dependencies to `code/requirements.txt` with pinned versions. The summarization module itself uses only the Python standard library.
- Read API keys from environment variables. `.env` is ignored by git and excluded from submission packages; this summarizer does not load `.env` files automatically.
- Keep the internal implementation instructions in each component's own README. This README describes the handoff between retrieval, the shared app, and the downstream modules.
- Update the downstream-task entry in the root `README.md` when the modules are integrated.
