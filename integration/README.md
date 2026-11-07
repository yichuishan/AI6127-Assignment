# Group App Integration

This folder contains the minimal summarization integration prepared for the group repository:

- `app.py`: replace the group's `code/app.py` with this file.
- `app.patch`: an alternative patch that changes only `code/app.py`.
- `test_app_generation.py`: optional integration tests; place at `code/tests/test_app_generation.py`.

The patch is based on group `main` commit `ab5844e`, which already includes summarization. It makes game selection mandatory, applies the selected-game filter to sparse and dense retrieval, checks candidates before fusion/reranking, and checks the final evidence again at the generation boundary.

## Apply

Either upload `app.py` to `code/app.py`, or download `app.patch` and apply it from the group repository root:

```bash
git apply --check /path/to/app.patch
git apply /path/to/app.patch
```

Use one of these methods. The group repository must already contain the updated `code/generation/rag_summary/`, including `review_context.py`. Live generation uses its latest-update context policy by default.

## Run

Set `GENERATION_API_KEY` in the terminal environment. Optional settings are `GENERATION_BASE_URL` (default `https://api.deepseek.com`) and `GENERATION_MODEL` (default `deepseek-v4-pro`). The fallback key is `OPENAI_API_KEY`. This integration does not automatically load `.env` files.

From the group repository's `code/` directory:

```bash
streamlit run app.py
```

Select at least one target game, enter the question, then click **Generate summary** after retrieval. For comparisons, select the specific games to compare. Empty game selection does not trigger retrieval or generation. Reviews with missing or out-of-scope game metadata are excluded.

Generation produces up to four claims using the evidence-quote configuration, shows citations and review links, and provides a JSON download. The latest successful result is saved to `data/generation_results/rag_summary/latest_answer.json` through the shared `paths.py`, including the applied `retrieval_filters`.

Normal reruns reuse the stored answer. A changed question, game scope, or evidence hides the previous answer. Clearing the game selection removes the stored answer. API/configuration errors leave the retrieval results usable. A local save error preserves the generated answer and its download.

## Verification

Verified with Streamlit `1.66.1` using its real `AppTest` runtime and a local HTTP API implementing Chat Completions. No paid model calls were made.

All 23 integration tests passed:

- Generation starts only when clicked; citations, environment settings and output files are correct.
- Reruns reuse the result without another API call.
- Changes to the question, evidence, or review update timestamps hide stale answers.
- Explicit review updates reach the model without earlier opinions; selected text and timestamps are recorded in the output.
- Helpful votes and publication-age weights reach the model and are saved with the shared reference time.
- All four retrieval modes and the existing game filter feed the summarizer.
- No game selection means no retrieval or model call; multiple explicitly selected games support comparisons.
- Out-of-scope candidates are excluded even when a retriever ignores its filter argument.
- The generation entry point also excludes foreign or unknown game metadata when called directly.
- Changing game scope hides the old answer even if the highest-ranked review stays the same.
- Empty retrieval, missing keys, HTTP errors, invalid JSON and invalid configuration are handled.
- Abstention can be saved/downloaded, and file-write failures preserve the displayed answer.

Tests execute the group's query expansion, fusion and reranking control flow, together with the actual summarizer and HTTP client. Sparse/dense scoring and the cross-encoder's model weights are replaced by fixtures. This verifies application integration rather than real retrieval rankings or model answer quality.

After placing the test file at the location above, run from `code/` using the group's dependencies:

```bash
python -m unittest discover -s tests -p test_app_generation.py -v
```
