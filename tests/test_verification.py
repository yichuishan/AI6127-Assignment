"""Scripted verdicts test orchestration, not real entailment accuracy."""
import json
import unittest

from rag_generator import generate_summary, GenerationError
from evaluate_generation import evaluate, judge_result

HITS = [
    {'chunk_id': 'r1', 'text': 'It runs smoothly on my PC, but crashes sometimes.',
     'metadata': {'source_url': 'https://example.com/r1', 'game_name': 'Example'}},
    {'chunk_id': 'r2', 'text': 'I get frequent frame drops.', 'metadata': None},
]
BAD = {'text': 'Players report stable performance.', 'chunk_ids': ['r1'],
       'evidence': [{'chunk_id': 'r1', 'quote': HITS[0]['text']}]}
FIXED = {'text': 'One reviewer reports smooth performance on their PC but occasional crashes.',
         'chunk_ids': ['r1'], 'evidence': BAD['evidence']}


def verdict(label):
    return {'verdict': label, 'reason': 'Scripted test judgment'}


class ScriptedClient:
    def __init__(self, *outputs, usage=True):
        self.outputs, self.calls, self.usage = list(outputs), [], usage

    def complete(self, messages, model, temperature):
        if not self.outputs:
            raise AssertionError('Unexpected extra model call')
        self.calls.append(messages)
        return {'content': json.dumps(self.outputs.pop(0)),
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20} if self.usage else {}}


class VerificationTests(unittest.TestCase):
    def test_structured_mode_does_not_verify_semantics(self):
        client = ScriptedClient({'claims': [BAD]})
        result = generate_summary('How is performance?', HITS, approach='improved', client=client)
        self.assertEqual(result['claims'][0]['text'], BAD['text'])
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(result['validation']['semantic_check'])

    def test_delete_all_non_supported_categories(self):
        for label in ['partially_supported', 'contradicted', 'unsupported']:
            with self.subTest(label=label):
                client = ScriptedClient({'claims': [BAD]}, verdict(label))
                result = generate_summary('How is performance?', HITS, approach='verify_delete', client=client)
                self.assertEqual(result['status'], 'abstained')
                self.assertEqual(result['citations'], [])
                self.assertEqual(result['claim_audit'][0]['verification']['verdict'], label)
                self.assertEqual(result['validation']['semantic_discarded_claims'], 1)
                self.assertEqual(len(client.calls), 2)

    def test_repair_reverified_and_all_calls_costed(self):
        client = ScriptedClient({'claims': [BAD]}, verdict('partially_supported'), {'claim': FIXED}, verdict('supported'))
        result = generate_summary('How is performance?', HITS, client=client, input_cost_per_million=1, output_cost_per_million=2)
        self.assertEqual(result['approach'], 'verify_revise')
        self.assertEqual(result['claims'], [FIXED])
        self.assertEqual(result['claim_audit'][0]['outcome'], 'revised')
        self.assertEqual([c['stage'] for c in result['stage_calls']], ['generate', 'verify', 'revise', 'reverify'])
        self.assertEqual(result['usage'], {'prompt_tokens': 400, 'completion_tokens': 80})
        self.assertAlmostEqual(result['estimated_cost_usd'], .00056)
        self.assertEqual(result['validation']['revised_claims'], 1)
        first = json.loads(client.calls[1][1]['content'])
        second = json.loads(client.calls[3][1]['content'])
        self.assertEqual(len(first['evidence']), 2)
        self.assertEqual(first['evidence'], second['evidence'])
        self.assertEqual(second['claim'], FIXED)

    def test_unsupported_repair_is_discarded_without_retry_loop(self):
        client = ScriptedClient({'claims': [BAD]}, verdict('unsupported'), {'claim': FIXED}, verdict('partially_supported'))
        result = generate_summary('performance?', HITS, client=client)
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(result['claim_audit'][0]['outcome'], 'revision_rejected')
        self.assertEqual(len(client.calls), 4)

    def test_null_and_fabricated_repairs_skip_reverification(self):
        forged = dict(FIXED, evidence=[{'chunk_id': 'r1', 'quote': 'No crashes ever.'}])
        for candidate in [None, forged, dict(FIXED, chunk_ids=['invented'])]:
            client = ScriptedClient({'claims': [BAD]}, verdict('contradicted'), {'claim': candidate})
            result = generate_summary('performance?', HITS, client=client)
            self.assertEqual(result['status'], 'abstained')
            self.assertEqual(len(client.calls), 3)

    def test_supported_original_is_kept_without_repair(self):
        client = ScriptedClient({'claims': [FIXED]}, verdict('supported'))
        result = generate_summary('performance?', HITS, client=client)
        self.assertEqual(result['claim_audit'][0]['outcome'], 'kept')
        self.assertEqual(result['validation']['retained_original_claims'], 1)
        self.assertEqual(len(client.calls), 2)

    def test_malformed_verifier_or_repair_fails_closed(self):
        sequences = [
            [{'claims': [BAD]}, {'verdict': 'maybe', 'reason': 'invalid'}],
            [{'claims': [BAD]}, {'verdict': 'supported'}],
            [{'claims': [BAD]}, verdict('unsupported'), {}],
        ]
        for outputs in sequences:
            with self.assertRaises(GenerationError):
                generate_summary('performance?', HITS, client=ScriptedClient(*outputs))

    def test_empty_and_invalid_provenance_skip_semantic_calls(self):
        self.assertEqual(generate_summary('q', [], client=ScriptedClient())['stage_calls'], [])
        client = ScriptedClient({'claims': [dict(BAD, chunk_ids=['unknown'])]})
        result = generate_summary('q', HITS, client=client)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result['validation']['rejected_claims'], 1)

    def test_claim_limit_bounds_call_fanout(self):
        with self.assertRaises(GenerationError):
            generate_summary('q', HITS, max_claims=1, client=ScriptedClient({'claims': [BAD, BAD]}))
        with self.assertRaises(ValueError):
            generate_summary('q', HITS, max_claims=0)

    def test_missing_usage_does_not_fabricate_cost(self):
        client = ScriptedClient({'claims': [FIXED]}, verdict('supported'), usage=False)
        result = generate_summary('q', HITS, client=client, input_cost_per_million=1, output_cost_per_million=2)
        self.assertEqual(result['usage'], {})
        self.assertIsNone(result['estimated_cost_usd'])

    def test_coverage_judge_schema(self):
        payload = {'faithfulness': {'score': 4, 'reason': 'test'}, 'relevance': {'score': 4, 'reason': 'test'},
                   'coverage': [{'covered': True, 'reason': 'test'}]}
        scored = judge_result('q', {'summary': 'Example'}, HITS, ScriptedClient(payload), 'mock', ['Crashes are reported'])
        self.assertTrue(scored['scores']['coverage'][0]['covered'])
        payload['coverage'] = []
        with self.assertRaises(GenerationError):
            judge_result('q', {'summary': 'Example'}, HITS, ScriptedClient(payload), 'mock', ['Crashes'])

    def test_ablation_aggregates_all_arms(self):
        outputs = [{'claims': [BAD]}, {'claims': [BAD]}, {'claims': [BAD]}, verdict('unsupported'),
                   {'claims': [BAD]}, verdict('partially_supported'), {'claim': FIXED}, verdict('supported')]
        result = evaluate([{'query': 'performance?', 'results': HITS}], client=ScriptedClient(*outputs))
        self.assertEqual(len(result['rows']), 4)
        self.assertEqual(result['aggregates']['verify_delete']['original_claim_retention_rate'], 0)
        self.assertEqual(result['aggregates']['verify_revise']['revised_claims'], 1)
        self.assertEqual(result['aggregates']['verify_revise']['mean_model_calls'], 4)
        self.assertIsNone(result['aggregates']['verify_revise']['key_point_coverage'])


if __name__ == '__main__':
    unittest.main()
