import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rag_generator import generate_summary, build_context, load_retrieval, GenerationError, OpenAICompatibleClient
from evaluate_generation import evaluate, judge_result

FIXTURE = Path(__file__).parent / 'fixtures/retrieval_export.json'


class FakeClient:
    def __init__(self, claims=None, raw=None):
        self.claims = claims if claims is not None else [{'text': 'One reviewer praises the story but reports unstable performance.', 'chunk_ids': ['review_001'], 'evidence': [{'chunk_id': 'review_001', 'quote': 'The story is excellent but performance is unstable.'}]}]
        self.raw, self.calls = raw, []

    def complete(self, messages, model, temperature):
        self.calls.append(messages)
        content = json.dumps({'claims': self.claims})
        if 'evidence verifier' in messages[0]['content']:
            content = json.dumps({'verdict': 'supported', 'reason': 'Mock judgment, not semantic validation.'})
        return {'content': self.raw if self.raw is not None else content,
                'usage': {'prompt_tokens': 100, 'completion_tokens': 40}}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.query, self.hits = load_retrieval(FIXTURE)

    def test_both_modes_and_source_provenance(self):
        before = copy.deepcopy(self.hits)
        for mode in ('baseline', 'improved'):
            client = FakeClient()
            result = generate_summary(self.query, self.hits, approach=mode, client=client)
            self.assertEqual(result['status'], 'ok')
            self.assertIn('[review_001]', result['summary'])
            self.assertEqual(result['citations'][0]['source_url'], 'https://example.com/review/001')
            self.assertIn('untrusted', client.calls[0][0]['content'])
        self.assertEqual(before, self.hits)

    def test_empty_does_not_call_model(self):
        client = FakeClient()
        self.assertEqual(generate_summary('question', [], client=client)['status'], 'no_evidence')
        self.assertFalse(client.calls)

    def test_missing_metadata(self):
        claim = {'text': 'One reviewer reports smooth performance.', 'chunk_ids': ['review_002'], 'evidence': [{'chunk_id': 'review_002', 'quote': 'It runs smoothly on my PC.'}]}
        result = generate_summary(self.query, self.hits, client=FakeClient([claim]))
        self.assertIsNone(result['citations'][0]['source_url'])

    def test_reject_unknown_ids_and_fabricated_quotes(self):
        for claim in [
            {'text': 'Invented', 'chunk_ids': ['unknown']},
            {'text': 'Invented', 'chunk_ids': ['review_001'], 'evidence': [{'chunk_id': 'review_001', 'quote': 'perfect performance'}]},
            {'text': 'Invented', 'chunk_ids': ['review_001', 'review_002'], 'evidence': [{'chunk_id': 'review_001', 'quote': 'The story'}]},
            {'text': 'Invented', 'chunk_ids': [{}]},
        ]:
            result = generate_summary(self.query, self.hits, client=FakeClient([claim]))
            self.assertEqual(result['status'], 'abstained')
            self.assertEqual(result['citations'], [])
            self.assertEqual(result['validation']['rejected_claims'], 1)

    def test_budget_deduplication_and_truncation(self):
        ctx = build_context(self.hits + self.hits, max_chunk_chars=10)
        self.assertEqual(len(ctx), 2)
        self.assertEqual(len(ctx[0]['text']), 10)
        self.assertLessEqual(len(json.dumps(build_context(self.hits, max_context_chars=200), ensure_ascii=False)), 200)
        result = generate_summary(self.query, self.hits, max_chunk_chars=10, client=FakeClient())
        self.assertEqual(result['status'], 'abstained')

    def test_invalid_inputs_and_model_json(self):
        for query, hits in [('', self.hits), ('q', {}), ('q', [{'text': 'no id'}])]:
            with self.assertRaises(ValueError):
                generate_summary(query, hits, client=FakeClient())
        with self.assertRaises(GenerationError):
            generate_summary(self.query, self.hits, client=FakeClient(raw='not json'))

    def test_cost_and_paired_evaluation(self):
        result = evaluate([{'query': self.query, 'results': self.hits}], client=FakeClient(), input_price=1, output_price=2)
        self.assertEqual(len(result['rows']), 4)
        self.assertAlmostEqual(result['rows'][0]['result']['estimated_cost_usd'], .00018)
        self.assertIsNone(result['aggregates']['baseline']['judge_scores'])
        self.assertEqual(result['rows'][0]['reference_metrics'], {})

    def test_judge_validation(self):
        result = generate_summary(self.query, self.hits, client=FakeClient())
        payload = {'faithfulness': {'score': 4, 'reason': 'Test-only judgment'}, 'relevance': {'score': 3, 'reason': 'Test-only judgment'}}
        judge = judge_result(self.query, result, self.hits, FakeClient(raw=json.dumps(payload)), 'mock-judge')
        self.assertEqual(judge['scores']['faithfulness']['score'], 4)
        payload['faithfulness']['score'] = 6
        with self.assertRaises(GenerationError):
            judge_result(self.query, result, self.hits, FakeClient(raw=json.dumps(payload)), 'mock-judge')

    def test_transport_response(self):
        from io import BytesIO
        response = BytesIO(json.dumps({'choices': [{'message': {'content': '{"claims":[]}'}}], 'usage': {}}).encode())
        with patch('rag_generator.urlopen', return_value=response) as request:
            out = OpenAICompatibleClient(api_key='test-key').complete([], 'test-model')
            self.assertEqual(json.loads(out['content']), {'claims': []})
            body = json.loads(request.call_args.args[0].data)
            self.assertEqual(body['temperature'], 0)
            self.assertEqual(body['model'], 'test-model')


if __name__ == '__main__':
    unittest.main()
