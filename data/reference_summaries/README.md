# Bilingual Reference Summary Review

This directory contains **100 assistant-drafted English reference answers**, with Chinese translations of all questions, answers, 172 evidence-linked sentences, and 19 priority-review notes. Drafts are based on each question's fixed review evidence and the existing query-specific scope annotations. A/B/C/D generated answers were not used to write them. No generation or translation API key was used.

The completed user review is available in [`reviewed/`](reviewed/README.md): **100 English-ready references**, comprising 98 unchanged approvals and two translated Chinese edits. The q03 edit includes an additional wording confirmation from the user. The web interface and `run_generation_experiments.py` now use these finalized references for ROUGE-1/2/L scoring.

Records in `drafts/` remain pending templates. Original review evidence remains verbatim English; Chinese claim explanations are reading aids, not additional evidence. Import `reviewed/reference_review_records.jsonl` into the existing page to restore the completed review, including both translations.

All 100 answers have been rewritten as coherent paragraphs. They lead with the response to the question, merge repeated information, and connect benefits, limitations and disagreements. The evidence section maps each sentence back to its sources, rather than treating the answer as a sequence of review abstracts.

Only relatively high-vote reviews receive wording such as “a highly upvoted review praises/criticizes …” / “一条高赞评论称赞／批评……”, where useful. The common rule is `helpful_votes >= max(10, nearest-rank 75th percentile of distinct context reviews)`. Exact counts remain in evidence details, not answer prose. Other evidence is synthesized without vote attribution. Fluency does not justify adding unsupported history, current popularity, recent improvements, new mod development, or unanimity.

## Files

- `drafts/review.html`: standalone bilingual review page, with side-by-side English/Chinese summaries, Chinese editing, evidence, browser progress, and JSONL export/import.
- `drafts/reference_drafts.jsonl`: all drafts, aligned translations, claim-to-evidence mappings, source texts, and context hashes.
- `drafts/reference_review.csv`: bilingual spreadsheet review sheet.
- `drafts/review.md`: bilingual drafts and source evidence in Markdown.
- `drafts/manifest.json`: source hashes, version metadata, counts, preparation method, and checks.
- `drafts/synthesis_revision.json`: the active synthesis policy, high-vote rule and revision provenance.

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

Nineteen draft questions are flagged for priority review. When starting from the draft templates, confirm each answer; the completed review can instead be restored from the file above. The evidence headings show recorded helpful votes; expand a review to see publication/update timestamps and its original text. Browser progress is saved locally; export a backup after each review session.

Version `bilingual_synthesis_v4` restores English-only, `bilingual_v2`, and `bilingual_votes_v3` records while preserving personal edits, reviewer names, and notes. Unedited old text is replaced by the revised answer. Earlier approvals return to **pending** so the rewritten answer can be checked again. Old records remain stored under their previous browser key. Imports from this version also check `review_metadata_sha256`.

## Export Fields

Every exported record retains `id`, `context_sha256`, `review_metadata_sha256`, `reference_policy`, the original English/Chinese drafts, reviewer details, decision, and notes.

For confirmed Chinese changes:

- `edited_draft_zh`: the reviewer's Chinese text.
- `edited_draft_en`: the existing English text, retained for comparison.
- `content_reviewed: true` and `review_language: "zh"`.
- `translation_status: "pending_translation"`.
- `reference: null` and `human_reviewed: false`: no finalized English scoring reference yet.

Only English-ready records have a nonempty `reference` and `human_reviewed: true`. The approved-reference export excludes untranslated Chinese changes. Chinese characters are not accepted in the finalized English field.

When importing translated records, `reference` is the translated English text, with `translated_reference_en` holding the same text, `translated_source_zh` holding the exact reviewed Chinese text, and `translated_by` identifying the translator. If either language changes afterward, the translation must be reconciled again. Untranslated Chinese edits remain excluded from English-reference exports.

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

These drafts retain the original fixed review texts and their `context_sha256`; supplemental helpful-vote/date metadata and high-vote flags are stored separately in `review_metadata` with its own hash. They do not change the archived fact-selection reference sets or reported scores. The current ROUGE runner provides the same review metadata to every generation method. Newest-update filtering requires its own matching references because this set preserves the full fixed review texts.

## Validation

The Python test suite checks source counts, exact quotes, vote provenance, aligned summaries, and agreement between the JSONL, HTML, CSV, and Markdown copies. An optional Node.js check exercises browser-state migration, bilingual edits, translation handling, and export/import without external packages:

```bash
python -m unittest discover -s tests -p test_reference_review.py -v
node tests/reference_review_ui.cjs
```

Run these commands from the summarizer module root.
