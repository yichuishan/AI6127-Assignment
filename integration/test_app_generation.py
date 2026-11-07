"""Streamlit interaction tests with real generation HTTP calls to a local fixture API.

Sparse/dense scoring and model weights are replaced by fixtures; fusion, query
expansion, reranking control flow, Streamlit and rag_generator are exercised.
Run from code/: python -m unittest discover -s tests -p test_app_generation.py -v
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

import streamlit as st
from streamlit.testing.v1 import AppTest
import paths

HITS = [
    {'chunk_id': 'fixture:review:1', 'text': 'The soundtrack is excellent.',
     'metadata': {'game_name': 'Example Game', 'source_url': 'https://example.com/review/1'}},
    {'chunk_id': 'fixture:review:2', 'text': 'The game occasionally crashes on my PC.',
     'metadata': {'game_name': 'Example Game', 'source_url': 'https://example.com/review/2'}},
    {'chunk_id': 'fixture:review:3', 'text': 'The controls feel responsive.',
     'metadata': {'game_name': 'Another Game', 'source_url': 'https://example.com/review/3'}},
]


class SparseFixture:
    method = 'sparse'

    def __init__(self, *args, **kwargs):
        self.chunks = copy.deepcopy(HITS)

    def retrieve(self, query, top_k=20, filters=None):
        if 'no-matching-reviews' in query:
            return []
        selected = self.chunks
        if filters:
            selected = [h for h in selected if h['metadata']['game_name'] in filters['game_name']]
        return [dict(copy.deepcopy(h), score=float(len(selected)-i), retrieval_method=self.method)
                for i, h in enumerate(selected[:top_k])]


class DenseFixture(SparseFixture):
    method = 'dense'


class CrossEncoderFixture:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs, **kwargs):
        return [float(len(pairs)-i) for i in range(len(pairs))]


def fixture_module(name, attribute, value):
    module = types.ModuleType(name)
    setattr(module, attribute, value)
    return module


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(payload)
        if self.server.scenario == 'unauthorized':
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"private-provider-detail"}')
            return
        evidence = json.loads(payload['messages'][1]['content'])['evidence'][0]
        claims = [] if self.server.scenario == 'abstain' else [{
            'text': 'One reviewer describes their experience.',
            'chunk_ids': [evidence['chunk_id']],
            'evidence': [{'chunk_id': evidence['chunk_id'], 'quote': evidence['text']}],
        }]
        content = 'not-json' if self.server.scenario == 'invalid-json' else json.dumps({'claims': claims})
        body = json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                           'usage': {'prompt_tokens': 30, 'completion_tokens': 20}}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AppGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modules = patch.dict(sys.modules, {
            'retrieval.sparse_retriever': fixture_module('retrieval.sparse_retriever', 'SparseRetriever', SparseFixture),
            'retrieval.dense_retriever': fixture_module('retrieval.dense_retriever', 'DenseRetriever', DenseFixture),
            'sentence_transformers': fixture_module('sentence_transformers', 'CrossEncoder', CrossEncoderFixture),
        })
        cls.modules.start()
        cls.api = ThreadingHTTPServer(('127.0.0.1', 0), APIHandler)
        threading.Thread(target=cls.api.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.api.shutdown()
        cls.api.server_close()
        cls.modules.stop()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name) / 'generation_results'
        self.api.requests = []
        self.api.scenario = 'success'
        env = patch.dict(os.environ, {'GENERATION_API_KEY': 'local-fixture-key', 'OPENAI_API_KEY': '',
             'GENERATION_BASE_URL': f'http://127.0.0.1:{self.api.server_port}/v1',
             'GENERATION_MODEL': 'fixture-model'})
        env.start(); self.addCleanup(env.stop)
        data = patch.multiple(paths, GENERATION_RESULTS_DIR=self.output,
                              QUERIES_PATH=Path(self.tmp.name) / 'no-sample-queries.jsonl')
        data.start(); self.addCleanup(data.stop)
        st.cache_data.clear()
        st.cache_resource.clear()

    def app(self, query='How is the soundtrack?'):
        app = AppTest.from_file(str(CODE / 'app.py'), default_timeout=20).run()
        self.assertFalse(app.exception)
        if query:
            app.text_input(key='query').set_value(query).run()
            self.assertFalse(app.exception)
        return app

    def generate(self, app):
        app.button(key='rag_summary_generate').click().run()
        self.assertFalse(app.exception)
        return app

    def answer_visible(self, app):
        return any('One reviewer describes' in item.value for item in app.markdown)

    def test_search_does_not_call_generation_until_clicked(self):
        app = self.app()
        self.assertEqual(self.api.requests, [])
        self.generate(app)
        self.assertEqual(len(self.api.requests), 1)
        self.assertTrue(self.answer_visible(app))
        result = json.loads((self.output / 'rag_summary/latest_answer.json').read_text())
        self.assertEqual(result['model'], 'fixture-model')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['citations'][0]['chunk_id'], 'fixture:review:1')
        self.assertEqual(result['citations'][0]['source_url'], HITS[0]['metadata']['source_url'])
        self.assertNotIn('local-fixture-key', json.dumps(result))
        payload = json.loads(self.api.requests[0]['messages'][1]['content'])
        self.assertEqual(payload['question'], 'How is the soundtrack?')
        self.assertEqual(payload['max_claims'], 4)
        self.assertNotIn('fact_candidates', payload)
        self.assertEqual(set(payload['evidence'][0]),
                         {'chunk_id', 'text', 'game_name', 'source_url', 'evidence_scope', 'is_high_helpfulness'})

    def test_latest_update_is_used_by_live_app_and_saved_for_inspection(self):
        updated = copy.deepcopy(HITS)
        updated[0]['text'] = 'Old: the soundtrack is poor.\nUpdate: The soundtrack is excellent.'
        updated[0]['metadata']['created_at_utc'] = '2026-01-01T00:00:00Z'
        updated[0]['metadata']['updated_at_utc'] = '2026-02-22T06:15:20Z'
        updated[0]['metadata']['helpful_votes'] = 120
        with patch(__name__ + '.HITS', updated):
            app = self.generate(self.app())
        self.assertTrue(self.answer_visible(app))
        payload = json.loads(self.api.requests[0]['messages'][1]['content'])
        self.assertEqual(payload['evidence'][0]['text'], 'Update: The soundtrack is excellent.')
        self.assertEqual(payload['evidence'][0]['updated_at_utc'], '2026-02-22T06:15:20+00:00')
        self.assertEqual(payload['evidence'][0]['helpful_votes'], 120)
        self.assertGreater(payload['evidence'][0]['helpful_votes_per_day'], 0)
        self.assertNotIn('soundtrack is poor', json.dumps(payload))
        result = json.loads((self.output / 'rag_summary/latest_answer.json').read_text())
        self.assertEqual(result['context_evidence'], payload['evidence'])
        self.assertEqual(result['context_as_of_utc'], payload['as_of_utc'])

    def test_changed_update_timestamp_hides_cached_answer(self):
        app = self.generate(self.app())
        updated = copy.deepcopy(HITS)
        updated[0]['metadata']['updated_at_utc'] = '2026-02-22T06:15:20Z'
        with patch(__name__ + '.HITS', updated):
            st.cache_resource.clear()
            st.cache_data.clear()
            app.run()
        self.assertFalse(app.exception)
        self.assertFalse(self.answer_visible(app))
        self.assertEqual(len(self.api.requests), 1)

    def test_rerun_reuses_answer_without_another_api_call(self):
        app = self.generate(self.app())
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(self.answer_visible(app))
        self.assertEqual(len(self.api.requests), 1)
        self.assertEqual(len(app.get('download_button')), 2)

    def test_changed_question_hides_old_answer(self):
        app = self.generate(self.app())
        app.text_input(key='query').set_value('How is performance?').run()
        self.assertFalse(self.answer_visible(app))
        self.assertEqual(len(self.api.requests), 1)
        self.generate(app)
        self.assertTrue(self.answer_visible(app))
        self.assertEqual(len(self.api.requests), 2)

    def test_changed_evidence_hides_old_answer(self):
        app = self.generate(self.app())
        app.sidebar.slider[0].set_value(1).run()
        self.assertFalse(app.exception)
        self.assertFalse(self.answer_visible(app))
        self.assertEqual(len(self.api.requests), 1)

    def test_all_four_retrieval_modes_connect(self):
        app = self.app()
        for mode in ['Hybrid + Rerank', 'Hybrid (RRF)', 'BM25', 'Dense']:
            with self.subTest(mode=mode):
                app.sidebar.radio[0].set_value(mode).run()
                self.generate(app)
                self.assertTrue(self.answer_visible(app))
        self.assertEqual(len(self.api.requests), 4)

    def test_game_filter_is_preserved(self):
        app = self.app()
        app.sidebar.multiselect(key='games').set_value(['Another Game']).run()
        self.generate(app)
        payload = json.loads(self.api.requests[0]['messages'][1]['content'])
        self.assertEqual([h['game_name'] for h in payload['evidence']], ['Another Game'])

    def test_no_matching_reviews_do_not_call_model(self):
        app = self.app('no-matching-reviews')
        self.assertTrue(app.warning)
        self.assertFalse(any(b.key == 'rag_summary_generate' for b in app.button))
        self.assertEqual(len(self.api.requests), 0)

    def test_missing_key_is_safe_and_retrieval_still_renders(self):
        os.environ.update(GENERATION_API_KEY='', OPENAI_API_KEY='', GENERATION_BASE_URL='https://api.deepseek.com')
        app = self.generate(self.app())
        self.assertTrue(app.error)
        self.assertEqual(len(self.api.requests), 0)
        self.assertTrue(any('Top 3 reviews' in item.value for item in app.subheader))

    def test_api_errors_do_not_leak_provider_details_or_show_stale_answers(self):
        app = self.generate(self.app())
        self.api.scenario = 'unauthorized'
        self.generate(app)
        self.assertTrue(app.error)
        self.assertFalse(self.answer_visible(app))
        self.assertNotIn('private-provider-detail', str([e.value for e in app.error]))

    def test_invalid_json_is_reported_without_crashing(self):
        self.api.scenario = 'invalid-json'
        app = self.generate(self.app())
        self.assertTrue(app.error)
        self.assertFalse(self.answer_visible(app))

    def test_invalid_api_configuration_is_reported_without_crashing(self):
        os.environ['GENERATION_BASE_URL'] = 'not-a-valid-url'
        app = self.generate(self.app())
        self.assertTrue(app.error)
        self.assertEqual(len(self.api.requests), 0)

    def test_abstention_remains_a_valid_downloadable_result(self):
        self.api.scenario = 'abstain'
        app = self.generate(self.app())
        self.assertFalse(app.error)
        result = json.loads((self.output / 'rag_summary/latest_answer.json').read_text())
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(result['claims'], [])
        self.assertEqual(len(app.get('download_button')), 2)

    def test_file_write_error_keeps_answer_available(self):
        self.output.write_text('This fixture blocks directory creation.')
        app = self.generate(self.app())
        self.assertTrue(app.warning)
        self.assertTrue(self.answer_visible(app))
        self.assertFalse(app.error)
        self.assertEqual(len(app.get('download_button')), 2)


if __name__ == '__main__':
    unittest.main()
