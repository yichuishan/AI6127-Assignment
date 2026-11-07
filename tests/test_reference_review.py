"""Integrity checks for bilingual, vote-aware reference drafts and review exports."""
import csv
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / 'data/reference_summaries/drafts'


class ReferenceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in (DRAFTS / 'reference_drafts.jsonl').read_text().splitlines()]
        cls.manifest = json.loads((DRAFTS / 'manifest.json').read_text())
        cls.audit = json.loads((DRAFTS / 'attribution_revision.json').read_text())
        cls.sources = {h['chunk_id']: h for line in (ROOT / 'experiments/steam_generation_eval.jsonl').read_text().splitlines()
                       for h in json.loads(line)['results']}

    def test_complete_pending_dataset_and_source_hashes(self):
        self.assertEqual(len(self.rows), 100)
        self.assertEqual(len({r['id'] for r in self.rows}), 100)
        self.assertEqual(sum(len(r['draft_claims']) for r in self.rows), 193)
        for r in self.rows:
            self.assertEqual(r['review_status'], 'pending')
            self.assertFalse(r['human_reviewed'])
            self.assertIsNone(r['reference'])
        for name, digest in self.manifest['source_files_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((DRAFTS / 'reference_drafts.jsonl').read_bytes()).hexdigest(),
                         self.manifest['reference_drafts_sha256'])

    def test_every_vote_count_and_quote_maps_to_original_source(self):
        for r in self.rows:
            for cid, metadata in r['review_metadata'].items():
                source = self.sources[cid]['metadata']
                self.assertEqual(metadata, {k: source[k] for k in
                                           ('helpful_votes', 'created_at_utc', 'updated_at_utc') if k in source})
            for c in r['draft_claims']:
                cited = {e['chunk_id']: e for e in c['supporting_evidence']}
                for e in cited.values():
                    self.assertEqual(self.sources[e['chunk_id']]['text'][e['start']:e['end']], e['quote'])
                    self.assertIn(e['chunk_id'], r['review_metadata'])
                for mention in c.get('attribution_vote_sources', []):
                    self.assertIn(mention['chunk_id'], cited)
                    self.assertEqual(mention['sample_id'], cited[mention['chunk_id']]['sample_id'])
                    self.assertEqual(mention['helpful_votes'], r['review_metadata'][mention['chunk_id']]['helpful_votes'])
                    self.assertGreater(mention['helpful_votes'], 0)
                    self.assertIn(str(mention['helpful_votes']), c['text'])
                    self.assertIn(str(mention['helpful_votes']), c['text_zh'])

    def test_revisions_are_traceable_and_bilingual_claims_form_summaries(self):
        self.assertEqual(self.audit['revised_records'], 48)
        self.assertEqual(self.audit['revised_claims'], 54)
        self.assertEqual(len(self.audit['changes']), 54)
        records = {r['id']: r for r in self.rows}
        for change in self.audit['changes']:
            r = records[change['id']]
            c = next(c for c in r['draft_claims'] if c['claim_id'] == change['claim_id'])
            for key in ('text', 'text_zh'):
                self.assertEqual(change[key], c[key])
                self.assertNotEqual(change['previous_' + key], change[key])
        for r in self.rows:
            self.assertEqual(r['draft_reference_summary'], ' '.join(c['text'] for c in r['draft_claims']))
            self.assertEqual(r['draft_reference_summary_zh'], ''.join(c['text_zh'] for c in r['draft_claims']))
            for c in r['draft_claims']:
                if len(c['supporting_evidence']) == 1:
                    cid = c['supporting_evidence'][0]['chunk_id']
                    if r['review_metadata'][cid].get('helpful_votes', 0) > 0:
                        self.assertIn('helpful vote', c['text'])
                        self.assertIn('有用投票', c['text_zh'])
                        self.assertNotRegex(c['text'], r'(?i)\b(?:one|a) reviewer\b')
                        self.assertNotIn('一位评论者', c['text_zh'])

    def test_html_csv_and_markdown_show_the_same_drafts(self):
        page = (DRAFTS / 'review.html').read_text()
        embedded = re.search(r'<script id="draft-data"[^>]*>(.*?)</script>', page, re.S)
        self.assertEqual(json.loads(embedded[1]), self.rows)
        with (DRAFTS / 'reference_review.csv').open(encoding='utf-8-sig', newline='') as f:
            sheet = list(csv.DictReader(f))
        self.assertEqual(len(sheet), 100)
        markdown = (DRAFTS / 'review.md').read_text()
        for r, item in zip(self.rows, sheet):
            self.assertEqual(item['id'], r['id'])
            for key in ('draft_reference_summary', 'draft_reference_summary_zh', 'review_metadata_sha256'):
                self.assertEqual(item[key], r[key])
            self.assertIn(r['draft_reference_summary'], markdown)
            self.assertIn(r['draft_reference_summary_zh'], markdown)
            digest = hashlib.sha256(json.dumps(r['review_metadata'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            self.assertEqual(digest, r['review_metadata_sha256'])


if __name__ == '__main__':
    unittest.main()
