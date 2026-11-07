"""Completed human review: translation alignment, provenance and evaluation export."""
import hashlib
import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1] / 'data/reference_summaries'


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class CompletedReferenceReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / 'reviewed'
        cls.raw = {r['id']: r for r in jsonl(cls.directory / 'submitted_review_records.jsonl')}
        cls.records = {r['id']: r for r in jsonl(cls.directory / 'reference_review_records.jsonl')}
        cls.approved = jsonl(cls.directory / 'approved_references.jsonl')
        cls.drafts = {r['id']: r for r in jsonl(ROOT / 'drafts/reference_drafts.jsonl')}
        cls.manifest = json.loads((cls.directory / 'manifest.json').read_text())

    def test_all_100_references_are_english_ready_and_match_source_versions(self):
        self.assertEqual(len(self.approved), 100)
        self.assertEqual(len({r['id'] for r in self.approved}), 100)
        self.assertEqual(set(self.records), set(self.drafts))
        self.assertEqual(self.manifest['pending_references'], 0)
        for r in self.approved:
            self.assertTrue(r['human_reviewed'])
            self.assertEqual(r['reference_language'], 'en')
            self.assertTrue(r['reference'].strip())
            self.assertIsNone(re.search(r'[\u3400-\u9fff]', r['reference']))
            self.assertEqual(r['reference'], self.records[r['id']]['reference'])
            self.assertEqual(r['reference_zh'], self.records[r['id']]['edited_draft_zh'])
            for field in ('context_sha256', 'review_metadata_sha256'):
                self.assertEqual(r[field], self.drafts[r['id']][field])
            visible = {e['chunk_id'] for e in self.drafts[r['id']]['fixed_context']}
            self.assertTrue(set(r['evidence_chunk_ids']) <= visible)

    def test_98_unchanged_human_approvals_are_preserved_exactly(self):
        unchanged = [r for r in self.raw.values() if r['translation_status'] == 'not_required']
        self.assertEqual(len(unchanged), 98)
        for r in unchanged:
            self.assertEqual(self.records[r['id']], r)

    def test_two_translations_match_the_reviewed_chinese_exactly(self):
        translations = [r for r in self.records.values() if r['translation_status'] == 'translated']
        self.assertEqual({r['id'] for r in translations}, {'hybrid_rerank/q03', 'hybrid_rerank/q04'})
        for r in translations:
            self.assertEqual(r['translated_source_zh'], r['edited_draft_zh'])
            self.assertEqual(r['translated_reference_en'], r['edited_draft_en'])
            self.assertEqual(r['reference'], r['translated_reference_en'])
            self.assertEqual(r['translated_by'], 'assistant')
            self.assertTrue(r['human_reviewed'])
        q04 = self.records['hybrid_rerank/q04']
        self.assertEqual(q04['edited_draft_zh'], self.raw[q04['id']]['edited_draft_zh'])
        self.assertIn('no frame generation', q04['reference'])
        self.assertNotIn('after fixes', q04['reference'])

    def test_q03_uses_explicitly_confirmed_wording_and_relevant_evidence(self):
        q03 = self.records['hybrid_rerank/q03']
        self.assertIn('大部分评论', self.raw[q03['id']]['edited_draft_zh'])
        self.assertNotIn('大部分评论', q03['edited_draft_zh'])
        self.assertIn('severe input latency', q03['reference'])
        self.assertIn('Xbox One to PC', q03['reference'])
        self.assertEqual(q03['adjudication_source'], 'user_confirmation_in_chat')
        r = next(r for r in self.approved if r['id'] == q03['id'])
        self.assertEqual(set(r['evidence_chunk_ids']),
                         {'steam:1091500:237091103:chunk:001', 'steam:1091500:237099475:chunk:001'})

    def test_manifest_hashes_preserve_the_submission_and_processed_artifacts(self):
        for name, expected in self.manifest['file_sha256'].items():
            self.assertEqual(hashlib.sha256((self.directory / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(self.manifest['submission_file_sha256'],
                         self.manifest['file_sha256']['submitted_review_records.jsonl'])
        self.assertEqual(hashlib.sha256((ROOT / 'drafts/reference_drafts.jsonl').read_bytes()).hexdigest(),
                         self.manifest['source_drafts_sha256'])


if __name__ == '__main__':
    unittest.main()
