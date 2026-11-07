"""Integrity checks for bilingual, vote-aware reference drafts and review exports."""
import csv
import hashlib
import json
from pathlib import Path
import re
import unittest

from review_context import high_helpfulness_threshold, review_key


ROOT = Path(__file__).resolve().parents[1]
DRAFTS = ROOT / 'data/reference_summaries/drafts'


class ReferenceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in (DRAFTS / 'reference_drafts.jsonl').read_text().splitlines()]
        cls.manifest = json.loads((DRAFTS / 'manifest.json').read_text())
        cls.audit = json.loads((DRAFTS / 'synthesis_revision.json').read_text())
        cls.sources = {h['chunk_id']: h for line in (ROOT / 'experiments/steam_generation_eval.jsonl').read_text().splitlines()
                       for h in json.loads(line)['results']}

    def test_complete_pending_dataset_and_source_hashes(self):
        self.assertEqual(len(self.rows), 100)
        self.assertEqual(len({r['id'] for r in self.rows}), 100)
        self.assertEqual(sum(len(r['draft_claims']) for r in self.rows), 172)
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
            votes = {review_key(self.sources[cid]): m['helpful_votes'] for cid, m in r['review_metadata'].items()}
            threshold = high_helpfulness_threshold(votes.values())
            self.assertEqual(r['high_helpful_votes_threshold'], threshold)
            for cid, metadata in r['review_metadata'].items():
                source = self.sources[cid]['metadata']
                expected = {k: source[k] for k in ('helpful_votes', 'created_at_utc', 'updated_at_utc') if k in source}
                expected['is_high_helpfulness'] = source.get('helpful_votes', -1) >= threshold
                self.assertEqual(metadata, expected)
            for c in r['draft_claims']:
                cited = {e['chunk_id']: e for e in c['supporting_evidence']}
                for e in cited.values():
                    self.assertEqual(self.sources[e['chunk_id']]['text'][e['start']:e['end']], e['quote'])
                    self.assertIn(e['chunk_id'], r['review_metadata'])
                for cid in c.get('high_helpfulness_sources', []):
                    self.assertIn(cid, cited)
                    self.assertTrue(r['review_metadata'][cid]['is_high_helpfulness'])
                self.assertEqual('highly upvoted' in c['text'].lower(), '高赞' in c['text_zh'])
                if 'highly upvoted' in c['text'].lower():
                    self.assertTrue(c['high_helpfulness_sources'])

    def test_revisions_are_traceable_and_bilingual_claims_form_summaries(self):
        self.assertEqual(self.audit['records_rewritten'], 100)
        self.assertEqual(self.audit['claims'], 172)
        self.assertEqual(self.audit['counts_in_answer_prose'], False)
        for r in self.rows:
            self.assertEqual(r['draft_reference_summary'], ' '.join(c['text'] for c in r['draft_claims']))
            self.assertEqual(r['draft_reference_summary_zh'], ''.join(c['text_zh'] for c in r['draft_claims']))
            self.assertNotRegex(r['draft_reference_summary'], r'\d+\s+helpful votes?')
            self.assertNotRegex(r['draft_reference_summary_zh'], r'\d+\s*(?:次有用投票|人点赞)')
            self.assertNotRegex(r['draft_reference_summary'], r'(?i)\b(?:one|another) reviewer\b')
            self.assertNotIn('一位评论者', r['draft_reference_summary_zh'])
            self.assertEqual(r['reference_policy'], 'coherent_synthesis_v1')

    def test_l4d_answer_synthesizes_evidence_without_adding_popularity_or_mod_development(self):
        r = next(r for r in self.rows if r['query_id'] == 'q06')
        self.assertEqual(len(r['draft_claims']), 1)
        self.assertGreater(len(r['draft_claims'][0]['supporting_evidence']), 1)
        self.assertIn('friends', r['draft_reference_summary'])
        self.assertIn('mods', r['draft_reference_summary'])
        self.assertNotRegex(r['draft_reference_summary'].lower(), r'new mods|classic|all players|recent reviews')

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
