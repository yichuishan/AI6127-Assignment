import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rag_generator import generate_summary, build_context, load_retrieval, GenerationError, OpenAICompatibleClient

FIXTURE = Path(__file__).parent / 'fixtures/retrieval_export.json'


class FakeClient:
    def __init__(self, claims=None, raw=None):
        self.claims = claims if claims is not None else [{'text': 'One reviewer praises the story but reports unstable performance.', 'chunk_ids': ['review_001']}]
        self.raw, self.calls = raw, []

    def complete(self, messages, model, temperature):
        self.calls.append(messages)
        content = json.dumps({'claims': self.claims})
        return {'content': self.raw if self.raw is not None else content,
                'usage': {'prompt_tokens': 100, 'completion_tokens': 40}}


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.query, self.hits = load_retrieval(FIXTURE)

    def test_single_call_and_source_provenance(self):
        before=copy.deepcopy(self.hits)
        client=FakeClient()
        result=generate_summary(self.query,self.hits,client=client)
        self.assertEqual(result['status'],'ok')
        self.assertIn('[review_001]',result['summary'])
        self.assertEqual(result['citations'][0]['source_url'],'https://example.com/review/001')
        self.assertIn('untrusted',client.calls[0][0]['content'])
        self.assertEqual(len(client.calls),1)
        self.assertEqual(result['approach'],'baseline')
        self.assertEqual(before,self.hits)

    def test_empty_does_not_call_model(self):
        client = FakeClient()
        self.assertEqual(generate_summary('question', [], client=client)['status'], 'no_evidence')
        self.assertFalse(client.calls)

    def test_missing_metadata(self):
        claim = {'text': 'One reviewer reports smooth performance.', 'chunk_ids': ['review_002']}
        result = generate_summary(self.query, self.hits, client=FakeClient([claim]))
        self.assertIsNone(result['citations'][0]['source_url'])

    def test_reject_invalid_claims_and_unknown_citation_ids(self):
        for claim in [None,{}, {'text':'Invented','chunk_ids':['unknown']},
                      {'text':'Invented','chunk_ids':[{}]}, {'text':'','chunk_ids':['review_001']},
                      {'text':'Invented','chunk_ids':[]}]:
            result=generate_summary(self.query,self.hits,client=FakeClient([claim]))
            self.assertEqual(result['status'],'abstained')
            self.assertEqual(result['citations'],[])
            self.assertEqual(result['validation']['rejected_claims'],1)

    def test_budget_deduplication_and_truncation(self):
        ctx = build_context(self.hits + self.hits, max_chunk_chars=10)
        self.assertEqual(len(ctx), 2)
        self.assertEqual(len(ctx[0]['text']), 10)
        self.assertLessEqual(len(json.dumps(build_context(self.hits, max_context_chars=200), ensure_ascii=False)), 200)
        result = generate_summary(self.query, self.hits, max_chunk_chars=10, client=FakeClient())
        self.assertTrue(all(len(e['text'])<=10 for e in result['context_evidence']))

    def test_invalid_inputs_and_model_json(self):
        for query, hits in [('', self.hits), ('q', {}), ('q', [{'text': 'no id'}])]:
            with self.assertRaises(ValueError):
                generate_summary(query, hits, client=FakeClient())
        with self.assertRaises(GenerationError):
            generate_summary(self.query, self.hits, client=FakeClient(raw='not json'))

    def test_generation_usage_and_cost(self):
        result = generate_summary(self.query, self.hits, client=FakeClient(),
                                  input_cost_per_million=1, output_cost_per_million=2)
        self.assertEqual(result['usage'], {'prompt_tokens': 100, 'completion_tokens': 40})
        self.assertAlmostEqual(result['estimated_cost_usd'], .00018)

    def test_invalid_output_schema_and_claim_limit_are_visible_errors(self):
        for raw,code in [('[]','invalid_json'),('{}','invalid_claims'),
                         ('{"claims":{}}','invalid_claims')]:
            with self.assertRaises(GenerationError) as error:
                generate_summary(self.query,self.hits,client=FakeClient(raw=raw))
            self.assertEqual(error.exception.code,code)
            self.assertEqual(error.exception.stage,'generate')
        claim={'text':'A review describes its experience.','chunk_ids':['review_001']}
        with self.assertRaises(GenerationError):
            generate_summary(self.query,self.hits,max_claims=1,client=FakeClient([claim,claim]))

    def test_missing_usage_does_not_fabricate_tokens_or_cost(self):
        class NoUsage(FakeClient):
            def complete(self,*args,**kwargs):
                response=super().complete(*args,**kwargs)
                response.pop('usage')
                return response
        client=NoUsage()
        result=generate_summary(self.query,self.hits,client=client,
            input_cost_per_million=1,output_cost_per_million=2)
        self.assertEqual(result['usage'],{})
        self.assertIsNone(result['estimated_cost_usd'])
        self.assertEqual(len(client.calls),1)

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
