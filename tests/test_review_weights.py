"""Deterministic helpfulness/age signals and their use across generation stages."""
import copy
from datetime import datetime, timezone
import json
import unittest

from rag_generator import build_context, generate_summary
from review_context import context_time
from test_verification import ScriptedClient, verdict


AS_OF = '2026-11-06T12:00:00Z'


def review(votes=100, created='2026-10-27T12:00:00Z', **metadata):
    return {'chunk_id': 'steam:1:1:chunk:001',
            'text': 'Originally it crashed.\nUpdate: It runs smoothly on Windows.',
            'metadata': dict(helpful_votes=votes, created_at_utc=created,
                             updated_at_utc='2026-11-05T12:00:00Z', **metadata)}


class ReviewWeightTests(unittest.TestCase):
    def test_votes_divided_by_days_since_publication_not_last_edit(self):
        source = review()
        before = copy.deepcopy(source)
        entry = build_context([source], as_of=AS_OF)[0]
        self.assertEqual(entry['helpful_votes'], 100)
        self.assertEqual(entry['review_age_days'], 10)
        self.assertEqual(entry['helpful_votes_per_day'], 10)
        self.assertEqual(entry['text'], 'Update: It runs smoothly on Windows.')
        self.assertEqual(source, before)

    def test_fresh_reviews_use_a_one_day_floor(self):
        for created, expected_age in [('2026-11-06T00:00:00Z', .5), (AS_OF, 0)]:
            entry = build_context([review(20, created)], as_of=AS_OF)[0]
            self.assertEqual(entry['review_age_days'], expected_age)
            self.assertEqual(entry['helpful_votes_per_day'], 20)

    def test_missing_invalid_and_future_publication_dates_have_no_weight(self):
        for created in (None, 'bad date', '2026-11-07T00:00:00Z'):
            entry = build_context([review(created=created)], as_of=AS_OF)[0]
            self.assertEqual(entry['helpful_votes'], 100)
            self.assertNotIn('review_age_days', entry)
            self.assertNotIn('helpful_votes_per_day', entry)

    def test_unknown_votes_are_distinct_from_zero(self):
        for votes in (None, -1, True, 1.5, float('inf'), 'bad count'):
            entry = build_context([review(votes)], as_of=AS_OF)[0]
            self.assertEqual(entry['review_age_days'], 10)
            self.assertNotIn('helpful_votes', entry)
            self.assertNotIn('helpful_votes_per_day', entry)
        entry = build_context([review(0)], as_of=AS_OF)[0]
        self.assertEqual(entry['helpful_votes'], 0)
        self.assertEqual(entry['helpful_votes_per_day'], 0)

    def test_steam_raw_count_and_timestamp_fields_are_supported(self):
        source = review()
        source['metadata'] = {'votes_up': '100', 'timestamp_created': 1793102400}
        entry = build_context([source], as_of=AS_OF)[0]
        self.assertEqual(entry['helpful_votes'], 100)
        self.assertEqual(entry['helpful_votes_per_day'], 10)

    def test_weights_do_not_reorder_or_combine_review_sources(self):
        low = review(1)
        high = review(1000)
        high['chunk_id'] = 'steam:1:2:chunk:001'
        context = build_context([low, high, high], as_of=AS_OF)
        self.assertEqual([e['chunk_id'] for e in context], [low['chunk_id'], high['chunk_id']])
        self.assertEqual([e['helpful_votes_per_day'] for e in context], [.1, 100])

    def test_same_review_chunks_keep_a_single_review_identity_and_vote_count(self):
        first, second = review(100), review(100)
        first['text'] = 'Update: It runs smoothly'
        second['text'] = ' on Windows.'
        second['chunk_id'] = 'steam:1:1:chunk:002'
        context = build_context([first, second], as_of=AS_OF)
        self.assertEqual(len(context), 2)
        self.assertEqual([e['helpful_votes'] for e in context], [100, 100])
        self.assertTrue(all(e['chunk_id'].startswith('steam:1:1:') for e in context))

    def test_generation_verification_and_revision_receive_identical_signals_and_clock(self):
        source = review(120)
        claim = {'text': 'A review with 120 helpful votes reports smooth performance on Windows.',
                 'chunk_ids': [source['chunk_id']],
                 'evidence': [{'chunk_id': source['chunk_id'], 'quote': 'It runs smoothly on Windows.'}]}
        client = ScriptedClient({'claims': [claim]}, verdict('partially_supported'),
                                {'claim': claim}, verdict('supported'))
        result = generate_summary('How does it run?', [source], client=client, as_of=AS_OF)
        self.assertEqual(result['status'], 'ok')
        self.assertIn('120 helpful votes', result['summary'])
        self.assertEqual(result['context_as_of_utc'], '2026-11-06T12:00:00+00:00')
        self.assertEqual(result['context_evidence'][0]['helpful_votes_per_day'], 12)
        for messages in client.calls:
            payload = json.loads(messages[1]['content'])
            self.assertEqual(payload['as_of_utc'], result['context_as_of_utc'])
            self.assertEqual(payload['evidence'], result['context_evidence'])
            self.assertIn('REVIEW HELPFULNESS RULE', messages[0]['content'])
            self.assertIn('not additional reviewers', messages[0]['content'])

    def test_full_review_benchmark_input_and_prompts_remain_unchanged(self):
        client = ScriptedClient({'claims': []})
        result = generate_summary('How does it run?', [review()], approach='baseline', client=client,
                                  context_policy='full_review')
        payload = json.loads(client.calls[0][1]['content'])
        self.assertEqual(set(payload['evidence'][0]), {'chunk_id', 'text', 'game_name', 'source_url'})
        self.assertNotIn('as_of_utc', payload)
        self.assertNotIn('REVIEW HELPFULNESS RULE', client.calls[0][0]['content'])
        self.assertNotIn('context_as_of_utc', result)

    def test_explicit_clock_normalizes_timezones_and_invalid_values_are_rejected(self):
        expected = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
        for value in (AS_OF, '2026-11-06T20:00:00+08:00', expected, expected.replace(tzinfo=None)):
            self.assertEqual(context_time(value), expected)
        for value in ('bad time', True, float('inf')):
            with self.assertRaises(ValueError):
                generate_summary('q', [review()], as_of=value, client=ScriptedClient())


if __name__ == '__main__':
    unittest.main()
