# Evidence-grounded RAG Summarization

Independent generation for AI6127. Python 3.10+; core generation and offline tests use only the standard library. No Retrieval or aspect-based sentiment analysis is reimplemented.

## Retrieval interface

Previously inspected `GSrihariS/RAG_ASPA_Game_Retrieval` at commit `72243c22dae2de3c17c2e433cf7e7283a3c0cfeb`:

- `app.py`: `search(...) -> (hits, timings)`; downloaded JSON contains `query` and `results` plus retrieval settings/timings.
- `hybrid_retriever.py`: `retrieve()` returns separate sparse/dense lists. Use the ranked output from `retrieve_fused()` or UI search for generation.
- `fusion.py`: deduplicates by original chunk_id and preserves text/metadata; adds RRF and per-method ranks/scores.
- `reranker.py`: returns ranked dictionaries retaining original chunk_id/text/metadata, with reranker score and retrieval_method.

Generation accepts that ranked list directly and ignores score differences; scores are not probabilities. Missing/null metadata is supported. Original nonempty string chunk_id and string text are required; IDs are never invented. Source URLs are copied only for HTTP(S), otherwise null. Input hits are never mutated.

## Configuration and usage

From the repository root:

```bash
# Set securely; never commit credentials:
export GENERATION_API_KEY='<your key>'
export GENERATION_MODEL='gpt-4o-mini'
# Optional: export GENERATION_BASE_URL='https://your-provider.example/v1'
python rag_generator.py search_results.json --approach verify_revise --output my_summary.json
python -m unittest discover -s tests -v
```

`OPENAI_API_KEY` is a fallback. The default endpoint is `https://api.openai.com/v1/chat/completions`. A local OpenAI-compatible server can use `http://localhost:8000/v1` and its model ID without a key. Model weights/serving are not installed. Remote endpoints require HTTPS. The provider must support Chat Completions, JSON object mode, temperature 0 and max_tokens (default 8192). Requests use inherited proxy/TLS trust, a 180-second timeout and no automatic paid retries. Invalid API/JSON output raises `GenerationError`, not a fabricated success.

```python
from rag_generator import generate_summary, load_retrieval
query, hits = load_retrieval('search_results.json')
answer = generate_summary(query, hits)  # default: verify_revise
print(answer['summary'])
```

Optional arguments include approach, model, client, max_context_chars, max_chunk_chars, max_claims, input_cost_per_million and output_cost_per_million. The injectable client exposes `complete(messages, model, temperature)` returning content (JSON string) and usage.

## Context and four approaches

Context preserves hit order, skips blank text, deduplicates IDs (first nonempty occurrence), includes game name/URL and truncates each chunk to 4,000 characters. Total serialized evidence defaults to 24,000 characters; packing stops at the first entry exceeding the budget. Only supplied truncated text may be cited. These character limits are not exact token limits; lower them for small model windows. Query is capped at 8,000 characters.

| Arm | approach | Method |
| --- | --- | --- |
| A | baseline | Evidence-only prompt, claim-level source IDs |
| B | improved | Structured claims with exact supporting quotes |
| C | verify_delete | B plus semantic verification; delete any claim not fully supported |
| D | verify_revise | B plus verification, one repair attempt, then reverification |

All arms reject nonexistent citation IDs. B/C/D also require an exact quote for every cited ID. A valid quote establishes provenance, not semantic support.

The verifier sees one claim, its citations and all the same retrieved context. It classifies supported, partially_supported, contradicted or unsupported, considering attribution, game identity, quantifiers, negation, conditions and disagreement. A correctly attributed individual opinion need not agree with other reviewers. `verify_delete` retains only supported claims. `verify_revise` attempts one evidence-grounded repair, which must pass quote checks and receive a fresh supported verdict. Invalid/null repairs or failed reverifications are discarded; no unlimited retry loop. Malformed verifier output fails explicitly.

Example: “It runs smoothly on my PC, but crashes sometimes” does not support “Players report stable performance.” A possible repair is “One reviewer reports smooth performance on their PC but occasional crashes.” Real model judgment accuracy must be measured, not inferred from the scripted example.

All stages use the same configured model/client for controlled comparison. For N provenance-valid draft claims, A/B use one call, C at most 1+N, D at most 1+3N. Default max_claims is 12 (configurable 1–50); over-limit drafts fail. Both CLIs accept `--max-claims` to limit fanout.

## Output

Outputs include query, summary, citations (original chunk_id/source_url), final claims, model, approach, status, latency_seconds, usage, estimated_cost_usd, context_chunk_ids, stage_calls, claim_audit and validation counts. Summary sentences include `[original_chunk_id]` markers. Citations are rebuilt from accepted final claims only.

Statuses: `ok` (accepted claims), `abstained` (evidence but no accepted claims), `no_evidence` (no usable context; no API call). Audit entries preserve original claims, verdicts/reasons, repair candidates, second verdicts and outcomes. Counts record provenance rejections, retained originals, repaired and semantically discarded claims.

Latency and usage include generation, verification, repair and reverification. Costs require explicit USD-per-million input/output token prices and complete provider usage for every stage; otherwise cost is null and incomplete total usage is not presented. Billing, caching and hidden provider charges can differ. External evaluator-judge usage is recorded separately.

## Streamlit integration

Make this checkout importable (including prompts/) in the existing Retrieval app. After its `hits, timings = search(...)` call, add a separate generation section; leave the teammate's sentiment analysis independent:

```python
from rag_generator import generate_summary, GenerationError
if st.button('Generate evidence-grounded summary'):
    try:
        with st.spinner('Generating and verifying...'):
            answer = generate_summary(query, hits)
        st.write(answer['summary'])
        st.json(answer['citations'])
    except GenerationError as error:
        st.error(str(error))
```

The button avoids calls on every rerun. Cache/session results by query and hit contents if needed, invalidating them on evidence changes. Render without unsafe HTML; use the citations array for source URLs. Prompts treat review/question content as untrusted, but do not guarantee prevention of prompt injection.

## Evaluation and ablation

Provide JSONL with query/results and optional id, human-written reference, and key_points grounded in exactly those retrieved passages:

```json
{"id":"q1","query":"How is performance?","results":[{"chunk_id":"r1","text":"It stutters on my PC.","metadata":{}}],"reference":"One retrieved reviewer reports stuttering.","key_points":["A reviewer reports stuttering."]}
```

This is a synthetic format illustration, not a labeled Steam evaluation case. Retrieval relevance annotations are not generation references. No labeled generation dataset is supplied.

```bash
# All four arms by default:
python evaluate_generation.py evaluation.jsonl --output evaluation_results.json
# Optional external semantic judge and coverage (when key_points are supplied):
python evaluate_generation.py evaluation.jsonl --judge-model gpt-4o-mini
# Select arms and provide your provider's current USD-per-million prices:
python evaluate_generation.py evaluation.jsonl --approaches improved verify_delete verify_revise --input-price 1 --output-price 2
# Optional reference metrics:
python -m venv /workspace/generation-eval-venv
/workspace/generation-eval-venv/bin/pip install -r requirements-generation-eval.txt
/workspace/generation-eval-venv/bin/python evaluate_generation.py evaluation.jsonl --reference-metrics rouge bertscore
```

ROUGE-1/2/L F1 and BERTScore F1 run only with references and requested flags; counts disclose coverage. BERTScore uses roberta-large on CPU, requiring substantial Hugging Face model downloads, RAM/disk and compatible PyTorch/Transformers. That optional path is unverified here. No reference/judge means no corresponding quality score.

Reports contain full outputs, model calls, latency, usage/cost, rejection/repair counts, abstentions, mean final claim count and original-claim retention (retained originals / provenance-valid drafts). Retention is a count proxy, not information coverage. With human-authored key_points and an external judge, coverage records which points survive in the final answer. Report coverage alongside faithfulness to detect systems improving faithfulness simply by deleting useful content. Human support-category annotations and manual coverage audits are recommended.

The judge scores faithfulness/relevance from 1–5 with reasons; its predictions are estimates, not ground truth. Generator verifier verdicts are not independent quality metrics. Prefer an independent judge model and human checks. Every arm uses the same model, question, context budget and temperature, but drafts are generated independently and temperature 0 does not ensure identical outputs. B/C/D share a structured draft prompt; A/B change both structure and quote requirements. Use repeated paired runs and randomize arm order externally for reliable latency comparisons; the script's fixed order does not establish statistical significance.

## Tests, examples and limitations

```bash
python -m unittest discover -s tests -v
python tests/make_mock_examples.py
```

The 21 offline tests cover both original interfaces and semantic workflow: all verdict categories, deletion, repair/reverification, null/forged repairs, bounded call counts, malformed responses, evidence budgets, missing metadata, all-stage cost totals, missing usage, judge coverage schema and four-arm aggregation. CLI no-evidence checks require no credentials.

`tests/fixtures/retrieval_export.json` is synthetic. `generation_results.json` contains four explicitly scripted mock examples plus an actual no-evidence result. It demonstrates control flow only; timings are local mock execution, not model latency. Regeneration uses test-only clients and never calls an API.

No real API inference/verification/repair or labeled generation scoring has been performed. Credentials and reference data are required for those checks. No quality gains or evaluation scores are claimed from mocks. Quote matching and LLM verification can both miss unsupported meaning; same-model verification may share generation errors. No Retrieval or independent sentiment module is changed.

## Reproducible report experiments

Use `run_generation_experiments.py` for the pinned Steam benchmark, shared-draft
ablation, resumable checkpoints, detailed independent judging, report exports,
and human annotation agreement. See [EXPERIMENTS.md](EXPERIMENTS.md) for private
API-key setup and commands. `experiments/steam_generation_eval.jsonl` contains
100 real queries with saved condensed benchmark evidence; references still need
human annotation. The prepared data and preflight files are not LLM results.


Web UI: `python launch_web.py` opens the experiment page on your own computer.
For cloud port preview, run `python experiment_web.py --host 0.0.0.0 --port 8080`.
The page keeps keys in memory, shows task/stage progress, supports stopping and
ZIP export, and uses the existing experiment runner. See EXPERIMENTS.md §9.


DeepSeek V4: use `https://api.deepseek.com` and `deepseek-v4-pro`. The web UI has a
preset and configurable output budget/timeout/thinking. Auto mode disables thinking
for official DeepSeek V4/Flash endpoints; other providers receive no thinking field.
HTTP status, empty/truncated outputs, network and schema errors are reported with
safe codes rather than raw provider messages. See EXPERIMENTS.md §10.
