# Reference Summary Drafts

This directory contains **100 assistant-drafted English reference summaries** for the fixed generation benchmark. They are based on the corresponding questions and original review evidence, with the existing query-specific scope annotations as guidance. A/B/C/D generated answers were not used to write these references. No generation API key was used.

All references start as **pending human review**. They are not yet confirmed gold references, and ROUGE has not been computed from them.

## Files

- `drafts/review.html`: standalone offline review page with editable summaries, per-claim sources, the full fixed context, progress tracking, and JSONL export/import.
- `drafts/reference_drafts.jsonl`: 100 draft records, 193 claim-to-evidence mappings, original source texts, and context hashes.
- `drafts/reference_review.csv`: spreadsheet review sheet with editable-reference and reviewer fields.
- `drafts/review.md`: readable drafts and source evidence.
- `drafts/manifest.json`: source hashes, counts, preparation method, and automated checks.

## Review

1. Download `review.html` and open it locally in a browser. No server or API key is needed.
2. Read each question, draft, cited sources, and the full fixed retrieval context. Check relevance, support, game identity, conditions, and useful disagreement.
3. Enter your reviewer name. Confirm the draft, edit and confirm it, or reject it. Click **Save this item** or **Save and next**.
4. Export all review records to share or resume the work. Export approved references when ready for ROUGE evaluation.

The interface is in Chinese; reference text is in English. Nineteen questions are flagged for priority review because of limited evidence, subtopic gaps, or important conditions. All 100 still require human confirmation.

Each exported record contains its query ID and context hash. Confirmed references have a nonempty `reference`, an explicit reviewer, and `human_reviewed: true`. Pending and rejected items are excluded from the approved-reference export. The original assistant draft remains recorded.

## Important source distinctions

- q03: a graphics compliment is not evidence of smooth PC performance.
- q12: the explicitly new Dawn of War player is not used as a longtime-fan reference.
- q51: an average-FPS report does not establish whether stuttering occurs.
- q53/q78: soundtrack or music details are not invented from generic sound praise.
- q76: old-PC reports about Resident Evil Requiem do not explain Left 4 Dead 2's performance.
- q77: the Los Santos visual comparison is explicitly about graphics mods.
- q10/q93: a later February-2026 edit is distinguished from the earlier performance report.

## Evaluation scope

These are 100 question-conditioned summary drafts, distinct from the 1,000 source review passages and their aspect labels. Approval should be based on the evidence rather than agreement with any one generation method. Keep reference text free of inline citation IDs; provenance is stored separately in the claim-evidence mappings.

After review, the same approved references can evaluate each method's saved summaries. No retrieval or answer regeneration is needed for the existing fixed-context experiment. References are valid only for the questions and contexts identified by their hashes.
