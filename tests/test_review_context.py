"""Temporal evidence selection, using real reviews and multi-chunk fixtures."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from rag_generator import build_context, generate_summary
from review_context import timestamp
from run_generation_experiments import BENCHMARK, benchmark, jsonl, run
from generation_experiment_design import select_cases
from evaluate_generation import rescore
from helpers import ScriptedClient
from test_rouge_generation import SummaryModel


def hit(text, cid='steam:1:1:chunk:001', **metadata):
    return {'chunk_id': cid, 'text': text, 'metadata': metadata}


class ReviewContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = jsonl(BENCHMARK)
        cls.reviews = {h['chunk_id']: h for c in cls.cases for h in c['results']}

    def test_solar_crown_latest_edit_only_with_original_citation_and_dates(self):
        source = self.reviews['steam:1249970:208994799:chunk:001']
        before = copy.deepcopy(source)
        context = build_context([source])
        self.assertEqual(context[0]['text'], source['text'][source['text'].index('edit:'):])
        self.assertNotIn('3070', context[0]['text'])
        self.assertNotIn('60 FPS', context[0]['text'])
        self.assertIn('not perfect', context[0]['text'])
        for field in ('chunk_id',):
            self.assertEqual(context[0][field], source[field])
        for field in ('source_url', 'created_at_utc', 'updated_at_utc'):
            self.assertEqual(context[0][field], source['metadata'][field])
        self.assertEqual(context[0]['evidence_scope'], 'latest_explicit_update')
        self.assertEqual(source, before)

    def test_repeated_and_numbered_edits_keep_latest_conditions(self):
        for cid, prefix, absent in [
            ('steam:2246340:235096566:chunk:001', 'Edit Edit:', '3 crashes'),
            ('steam:952070:66912827:chunk:001', '(Edit 2:', 'dedicated servers'),
        ]:
            source = self.reviews[cid]
            text = build_context([source])[0]['text']
            self.assertEqual(text, source['text'][source['text'].index(prefix):])
            self.assertNotIn(absent, text)
        text = build_context([self.reviews['steam:2246340:235096566:chunk:001']])[0]['text']
        self.assertIn('on windows', text)

    def test_administrative_edits_do_not_replace_substantive_review(self):
        source = self.reviews['steam:282560:38795062:chunk:001']
        self.assertEqual(build_context([source])[0]['text'], source['text'])
        text = 'Old opinion.\nUpdate: New experience.\nedit: grammar only'
        self.assertEqual(build_context([hit(text)])[0]['text'], 'Update: New experience.\n')

    def test_latest_snapshot_wins_even_with_lower_retrieval_rank(self):
        old = hit('Old snapshot.', updated_at_utc='2025-01-01Z')
        new = hit('New snapshot.', updated_at_utc='2026-02-22T06:15:20Z')
        other = hit('Another author.', 'steam:1:2:chunk:001')
        context = build_context([old, other, new])
        self.assertEqual([c['text'] for c in context], ['Another author.', 'New snapshot.'])

    def test_latest_snapshot_preserves_all_its_chunks(self):
        items = [hit('Old first.', updated_at_utc='2025-01-01'),
                 hit('Current first.', updated_at_utc='2026-01-01'),
                 hit('Current second.', 'steam:1:1:chunk:002', updated_at_utc='2026-01-01')]
        self.assertEqual([c['text'] for c in build_context(items)], ['Current first.', 'Current second.'])

    def test_review_identity_does_not_collapse_different_games(self):
        items = [hit('Game A.', 'a', appid=1, review_id='r'),
                 hit('Game B.', 'b', appid=2, review_id='r')]
        self.assertEqual(len(build_context(items)), 2)

    def test_dates_handle_reverse_ordered_update_blocks(self):
        for text, expected in [
            ('Update 2026-02-22: New experience.\nUpdate 2025-11-12: Old experience.',
             'Update 2026-02-22: New experience.\n'),
            ('Update: As of 02/2026, new experience.\nUpdate: As of 11/2025, old experience.',
             'Update: As of 02/2026, new experience.\n'),
            ('Update: This was broken in 2026.\nUpdate: It works now.', 'Update: It works now.'),
        ]:
            self.assertEqual(build_context([hit(text)])[0]['text'], expected)

    def test_update_heading_at_chunk_end_keeps_next_chunk(self):
        for offsets in (False, True):
            first = 'Old experience.\nUpdate:'
            second = '\nNow stable on Windows.'
            items = [hit(first, chunk_index=1), hit(second, 'steam:1:1:chunk:002', chunk_index=2)]
            if offsets:
                items[0]['metadata']['char_start'] = 0
                items[1]['metadata']['char_start'] = len(first)
            context = build_context(items[::-1])
            # Retrieval order is retained, while text boundaries follow chunk order.
            self.assertEqual([c['text'] for c in context], [second, 'Update:'])
            self.assertTrue(all(c['evidence_scope'] == 'latest_explicit_update' for c in context))

    def test_overlapping_chunks_remove_old_body_but_keep_update_continuation(self):
        text = 'Old experience.\nUpdate: Still broken.\nEdit 2: Now stable on Windows. More detail.'
        start = text.index('Edit 2:')
        items = [hit(text[:start + 30], char_start=0),
                 hit(text[start - 4:], 'steam:1:1:chunk:002', char_start=start - 4)]
        for entry in build_context(items):
            self.assertNotIn('Still broken', entry['text'])
            self.assertTrue(text[start:].startswith(entry['text']))

    def test_empty_update_does_not_erase_existing_evidence(self):
        for text in ('Original opinion.\nUpdate:', 'Original.\nUpdate: Recent experience.\nEdit:'):
            context = build_context([hit(text)])
            self.assertTrue(context)
            self.assertIn('Original opinion.' if 'opinion' in text else 'Recent experience.', context[0]['text'])

    def test_timestamp_only_edit_retains_latest_whole_snapshot(self):
        source = hit('Current full text.', created_at_utc='2025-01-01', updated_at_utc='2026-01-01')
        context = build_context([source])[0]
        self.assertEqual(context['text'], source['text'])
        self.assertEqual(context['evidence_scope'], 'latest_available_snapshot')

    def test_missing_and_invalid_dates_preserve_text_and_do_not_invent_dates(self):
        for value in (None, True, float('inf'), 'bad date', 10**40):
            self.assertIsNone(timestamp(value))
            context = build_context([hit('Review text.', updated_at_utc=value)])[0]
            self.assertEqual(context['text'], 'Review text.')
            self.assertNotIn('updated_at_utc', context)
        self.assertEqual(timestamp(0).isoformat(), '1970-01-01T00:00:00+00:00')

    def test_update_selected_before_context_truncation(self):
        text = 'Old text. ' * 1000 + '\nUpdate: Current experience.'
        self.assertEqual(build_context([hit(text)], max_chunk_chars=30)[0]['text'],
                         'Update: Current experience.')

    def test_single_generation_uses_only_filtered_text(self):
        cid='steam:1:1:chunk:001'
        source=hit('Old: 20 FPS.\nUpdate: Now stable on Windows.')
        claim={'text':'One reviewer reports stability on Windows.','chunk_ids':[cid]}
        client=ScriptedClient({'claims':[claim]})
        result=generate_summary('Performance?',[source],client=client)
        self.assertEqual(result['status'],'ok')
        self.assertEqual(len(client.calls),1)
        self.assertEqual(result['context_policy'],'latest_update')
        messages=client.calls[0]
        self.assertNotIn('20 FPS',messages[1]['content'])
        self.assertIn('TEMPORAL EVIDENCE RULE',messages[0]['content'])
        self.assertEqual(json.loads(messages[1]['content'])['evidence'],result['context_evidence'])

    def test_full_review_policy_preserves_fixed_context_contract(self):
        source = self.reviews['steam:1249970:208994799:chunk:001']
        context = build_context([source], context_policy='full_review')
        self.assertEqual(context, [{'chunk_id': source['chunk_id'], 'text': source['text'],
                                   'game_name': source['metadata']['game_name'],
                                   'source_url': source['metadata']['source_url']}])
        client = ScriptedClient({'claims': []})
        result = generate_summary('Performance?', [source], client=client,
                                  context_policy='full_review')
        self.assertNotIn('TEMPORAL EVIDENCE RULE', client.calls[0][0]['content'])
        self.assertEqual(result['context_evidence'],context)

    def test_benchmark_uses_full_reviews_and_rejects_latest_update_rescoring(self):
        case=next(c for c in select_cases(self.cases) if any('208994799' in h['chunk_id'] for h in c['results']))
        benchmark([case])
        with tempfile.TemporaryDirectory() as tmp:
            model=SummaryModel()
            rows=run([case],Path(tmp)/'run',client=model,model='scripted',approaches=['baseline'])
            self.assertEqual(rows[0]['result']['context_policy'],'full_review')
            self.assertIn('60 FPS',model.calls[0][1]['content'])
            rows[0]['result']['context_policy']='latest_update'
            with self.assertRaises(ValueError):rescore(rows)

    def test_group_namespace_import_can_find_temporal_helper(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root.parent))
        try:
            module = importlib.import_module(root.name + '.rag_generator')
            self.assertEqual(module.build_context([hit('Old.\nEdit: New.')])[0]['text'], 'Edit: New.')
        finally:
            sys.path.pop(0)


if __name__ == '__main__':
    unittest.main()
