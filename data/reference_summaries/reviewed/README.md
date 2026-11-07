# Human-reviewed Reference Summaries

This directory contains **100 finalized English reference summaries**, with the reviewed Chinese versions and evidence provenance. The submitted review marked 98 drafts approved and edited two Chinese answers. Both edits have been translated; q03 also incorporates the user's explicit confirmation of evidence-grounded wording in chat.

## Files

| File | Purpose |
|---|---|
| `approved_references.jsonl` | The 100 English scoring references, reviewed Chinese text, source hashes and evidence IDs |
| `reference_review_records.jsonl` | Completed review records; import this file into the existing bilingual review page to restore 100 English-ready references |
| `submitted_review_records.jsonl` | The user's original export, preserved byte for byte |
| `adjudications.json` | The two translations and the q03 wording decision, including source IDs |
| `manifest.json` | Counts, source revision, file hashes and processing checks |

Each scoring record has `id`, `query`, `reference`, `reference_zh`, `reference_language`, `human_reviewed`, `context_sha256`, `review_metadata_sha256`, and `evidence_chunk_ids`, with reviewer and version fields. `reference` is the finalized English text. Join records to generated answers by question ID, not file order.

## Edited Answers

**q03 — Cyberpunk 2077 PC performance.** The final answer reports differing experiences: severe input latency even at the lowest settings, alongside a move from Xbox One to PC without a repeat of the prologue-ending crash. The user explicitly accepted this wording instead of claiming that most recent reviews establish stable, smooth performance.

**q04 — Monster Hunter Wilds frame rate.** The translation preserves the user's selected examples: 50–60 FPS on a low-end PC with drops in the Plains, around 50 FPS with balanced DLSS and no frame generation, and smooth laptop play. Information omitted by the user was not restored.

## Scope

These references apply to the existing 100 fixed questions and full review contexts, with the corresponding helpful-vote metadata. They are ready to use as English references for summary evaluation, including ROUGE. ROUGE has not yet been calculated from this completed review. Filtering reviews, changing passages or changing questions requires matching reference updates.

The existing fact-ID reference sets and archived experiment scores are separate and remain unchanged. No model API calls were made to finalize these records.
