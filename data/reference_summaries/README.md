# Bilingual Reference Summary Review

This directory contains **100 assistant-drafted English reference summaries**, with Chinese translations of all questions, draft summaries, 193 claim explanations, and 19 priority-review notes. Drafts are based on each question's fixed review evidence and the existing query-specific scope annotations. A/B/C/D generated answers were not used to write them. No generation or translation API key was used.

All source records remain **pending human review**. They are not confirmed gold references, and ROUGE has not been computed from them. Original review evidence remains verbatim English; Chinese claim explanations are reading aids, not additional evidence.

## Files

- `drafts/review.html`: standalone bilingual review page, with side-by-side English/Chinese summaries, Chinese editing, evidence, browser progress, and JSONL export/import.
- `drafts/reference_drafts.jsonl`: all drafts, aligned translations, claim-to-evidence mappings, source texts, and context hashes.
- `drafts/reference_review.csv`: bilingual spreadsheet review sheet.
- `drafts/review.md`: bilingual drafts and source evidence in Markdown.
- `drafts/manifest.json`: source hashes, version metadata, counts, preparation method, and checks.

## Review in Chinese

1. Open `drafts/review.html` locally in a browser. No server or API key is needed.
2. Read the Chinese question and summary alongside the English originals. Review each claim and expand its original sources or the complete fixed context.
3. Enter your reviewer name. The default editing language is Chinese; edit the Chinese summary as needed and add notes in Chinese.
4. Confirm the current content, confirm after edits, or reject it. Click **Save** or **Save & next**.
5. Export **all review records** or **Chinese edits awaiting translation**, and return the JSONL file to the assistant for an evidence-grounded English translation.

The page does not translate edits automatically or call an API. It keeps both languages and tracks the translation step separately:

| Decision | English-reference handling |
|---|---|
| Confirm the original draft without changing either language | The original English draft can be exported as approved |
| Confirm changes to the Chinese summary | Content is reviewed, but the English reference remains unavailable until translated |
| Edit English directly, leaving the Chinese draft unchanged | The reviewed English text can be exported; the Chinese text is labeled as the original comparison |
| Pending or rejected | Excluded from approved-reference export |

Nineteen questions are flagged for priority review. All 100 still require human confirmation. Browser progress is saved locally; export a backup after each review session. The page can restore review records from the previous English-only version and migrate its saved browser state.

## Export Fields

Every exported record retains `id`, `context_sha256`, the original English/Chinese drafts, reviewer details, decision, and notes.

For confirmed Chinese changes:

- `edited_draft_zh`: the reviewer's Chinese text.
- `edited_draft_en`: the existing English text, retained for comparison.
- `content_reviewed: true` and `review_language: "zh"`.
- `translation_status: "pending_translation"`.
- `reference: null` and `human_reviewed: false`: no finalized English scoring reference yet.

Only English-ready records have a nonempty `reference` and `human_reviewed: true`. The approved-reference export excludes untranslated Chinese changes. Chinese characters are not accepted in the finalized English field.

When importing translated records, `reference` is the translated English text, with `translated_reference_en` holding the same text, `translated_source_zh` holding the exact reviewed Chinese text, and `translated_by` identifying the translator. If either language changes afterward, the translation must be reconciled again. Review version `bilingual_v2` prevents the previous page from silently treating untranslated edits as English references.

## Important Source Distinctions

- q03: a graphics compliment is not evidence of smooth PC performance.
- q12: the explicitly new Dawn of War player is not used as a longtime-fan reference.
- q51: an average-FPS report does not establish whether stuttering occurs.
- q53/q78: radio or music details are not invented from general soundtrack/sound praise.
- q76: old-PC reports about Resident Evil Requiem do not explain Left 4 Dead 2's performance.
- q77: the Los Santos comparison is explicitly about graphics mods.
- q10/q93: a February-2026 edit is distinguished from the earlier performance report.

## Evaluation Scope

These are 100 question-conditioned summary drafts, distinct from the 1,000 source review passages and their aspect labels. Approval should be based on evidence rather than agreement with a generation method. Keep scoring reference text free of inline citation IDs; provenance is stored separately.

The same finalized English references can evaluate every method's saved summaries. Existing fixed-context experiments do not need retrieval or generation to run again. References apply only to the questions and contexts identified by their hashes.
