import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from rag_generator import OpenAICompatibleClient, GenerationError, safe_error


class DeepSeekTransportTests(unittest.TestCase):
    def response(self, content='{"claims":[]}', finish='stop'):
        return io.BytesIO(json.dumps({'choices':[{'message':{'content':content},'finish_reason':finish}],
                                     'usage':{'prompt_tokens':10,'completion_tokens':5}}).encode())

    def test_v4_auto_disables_thinking_and_increases_budget(self):
        with patch('rag_generator.urlopen',return_value=self.response()) as call:
            OpenAICompatibleClient(base_url='https://api.deepseek.com',api_key='test-key').complete([], 'deepseek-v4-pro')
        request=call.call_args.args[0]
        body=json.loads(request.data)
        self.assertEqual(request.full_url,'https://api.deepseek.com/chat/completions')
        self.assertEqual(body['thinking'],{'type':'disabled'})
        self.assertEqual(body['max_tokens'],8192)
        self.assertEqual(call.call_args.kwargs['timeout'],180)

    def test_other_providers_do_not_receive_deepseek_parameters(self):
        with patch('rag_generator.urlopen',return_value=self.response()) as call:
            OpenAICompatibleClient(base_url='https://example.com/v1',api_key='test-key').complete([], 'other-model')
        self.assertNotIn('thinking',json.loads(call.call_args.args[0].data))

    def test_explicit_thinking_and_budget(self):
        with patch('rag_generator.urlopen',return_value=self.response()) as call:
            OpenAICompatibleClient(base_url='https://api.deepseek.com',api_key='test-key',thinking='enabled',max_tokens=16000,timeout=300).complete([], 'deepseek-v4-pro')
        self.assertEqual(json.loads(call.call_args.args[0].data)['thinking'],{'type':'enabled'})
        self.assertEqual(json.loads(call.call_args.args[0].data)['max_tokens'],16000)

    def test_http_status_is_useful_but_body_and_key_are_not_exposed(self):
        for status in (400,401,402,403,404,422,429,500,503):
            error=HTTPError('https://example.com',status,'secret-raw-message',{},io.BytesIO(b'secret-provider-body'))
            with patch('rag_generator.urlopen',side_effect=error):
                with self.assertRaises(GenerationError) as caught:
                    OpenAICompatibleClient(api_key='secret-key').complete([], 'model')
            detail=safe_error(caught.exception)
            self.assertEqual(detail['http_status'],status)
            self.assertNotIn('secret',json.dumps(detail))
            self.assertNotIn('secret',str(caught.exception))

    def test_truncation_and_empty_json_mode_content(self):
        for content,finish,code in [('','length','output_truncated'),('  ','stop','empty_content'),(None,'stop','empty_content')]:
            with patch('rag_generator.urlopen',return_value=self.response(content,finish)):
                with self.assertRaises(GenerationError) as caught:OpenAICompatibleClient(api_key='test').complete([], 'model')
            self.assertEqual(safe_error(caught.exception)['error_code'],code)

    def test_network_timeout_is_distinguished(self):
        for error,code in [(URLError('raw-secret'), 'network_error'),(TimeoutError(),'timeout')]:
            with patch('rag_generator.urlopen',side_effect=error):
                with self.assertRaises(GenerationError) as caught:OpenAICompatibleClient(api_key='test').complete([], 'model')
            self.assertEqual(safe_error(caught.exception)['error_code'],code)
            self.assertNotIn('raw-secret',json.dumps(safe_error(caught.exception)))

if __name__=='__main__':unittest.main()
