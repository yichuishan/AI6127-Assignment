import http.cookiejar
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile

from experiment_web import Application, endpoint, server


class Model:
    def __init__(self, **kwargs):
        self.key = kwargs['api_key']
    def complete(self, messages, model, temperature):
        payload = json.loads(messages[1]['content'])
        system = messages[0]['content']
        if 'Evaluate an answer' in system:
            raise AssertionError('No evaluator model may be called by this webpage')
        elif 'evidence verifier' in system:
            obj={'verdict':'supported','reason':'Scripted test only.'}
        else:
            evidence=payload['evidence'][0]
            obj={'claims':[{'text':'A retrieved reviewer shares an opinion.', 'chunk_ids':[evidence['chunk_id']],
                'fact_ids':[c['fact_id'] for c in payload.get('fact_candidates',[]) if c['chunk_id']==evidence['chunk_id']][:1],
                'evidence':[{'chunk_id':evidence['chunk_id'],'quote':evidence['text'][:40]}]}]}
        return {'content':json.dumps(obj),'usage':{'prompt_tokens':10,'completion_tokens':5}}


class WebTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.app=Application(Path(self.tmp.name)/'runs', factory=Model)
        self.server=server('127.0.0.1',0,self.app)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)
        self.url=f'http://127.0.0.1:{self.server.server_port}'
        self.client=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.client.open(self.url).read()
    def post(self,path,data,client=None):
        req=urllib.request.Request(self.url+path,json.dumps(data).encode(),{'Content-Type':'application/json','Origin':self.url})
        return json.loads((client or self.client).open(req).read())
    def config(self):
        return {'api_key':'secret-web-test-only','base_url':'https://api.example.com/v1/chat/completions',
                'model':'scripted','limit':1,'repeats':1,'max_claims':2}
    def wait(self,jid):
        for _ in range(200):
            state=json.loads(self.client.open(self.url+'/api/jobs/'+jid).read())
            if state['status']!='running':return state
            time.sleep(.02)
        self.fail('Job did not finish')
    def test_real_http_job_progress_download_and_secret_exclusion(self):
        job=self.post('/api/jobs',self.config());state=self.wait(job['id'])
        self.assertEqual(state['status'],'completed');self.assertEqual(state['completed'],4)
        self.assertEqual(state['failed'],0);self.assertGreater(state['api_calls'],0)
        raw=self.client.open(self.url+'/api/jobs/'+job['id']+'/download').read()
        z=zipfile.ZipFile(io.BytesIO(raw))
        self.assertIn('results/summary.csv',z.namelist())
        self.assertIn('results/report.md',z.namelist())
        self.assertIn('results/predictions.jsonl',z.namelist())
        self.assertIn('results/evaluation_references.jsonl',z.namelist())
        rows=[json.loads(x) for x in z.read('results/checkpoint.jsonl').splitlines()]
        self.assertTrue(all('selected_fact_ids' in r['result'] for r in rows))
        self.assertTrue(all('fact_selection_f1' in r['metrics'] for r in rows))
        self.assertTrue(all('faithfulness' not in r['metrics'] for r in rows))
        for name in z.namelist():self.assertNotIn(b'secret-web-test-only',z.read(name))
        other=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        other.open(self.url).read()
        with self.assertRaises(urllib.error.HTTPError) as error:other.open(self.url+'/api/jobs/'+job['id'])
        self.assertEqual(error.exception.code,404)
    def test_validation_cross_origin_and_private_endpoints(self):
        for url in ['http://api.example.com/v1','https://127.0.0.1/v1','https://user:pass@api.example.com/v1','https://api.example.com/v1?key=secret']:
            with self.assertRaises(ValueError):endpoint(url)
        self.assertEqual(endpoint('https://api.example.com/v1/chat/completions'),'https://api.example.com/v1')
        body=self.config();body['limit']=0
        with self.assertRaises(urllib.error.HTTPError) as error:self.post('/api/jobs',body)
        self.assertEqual(error.exception.code,400)
        req=urllib.request.Request(self.url+'/api/jobs',json.dumps(self.config()).encode(),{'Content-Type':'application/json','Origin':'https://unrelated.example'})
        with self.assertRaises(urllib.error.HTTPError) as error:self.client.open(req)
        self.assertEqual(error.exception.code,403)
    def test_safe_diagnostic_reaches_progress_and_checkpoint(self):
        from rag_generator import GenerationError
        class NoBalance(Model):
            def complete(self,*a,**kw):
                raise GenerationError('raw-secret-not-for-display',code='http_402',http_status=402)
        self.app.factory=NoBalance
        cfg=self.config();cfg['approaches']=['baseline']
        job=self.post('/api/jobs',cfg);state=self.wait(job['id'])
        event=state['events'][0]
        self.assertEqual(event['http_status'],402)
        self.assertEqual(event['error_stage'],'generate')
        self.assertEqual(event['error_code'],'http_402')
        self.assertIn('余额',event['error_message'])
        self.assertNotIn('raw-secret',json.dumps(state))
        cfg2=self.app.jobs[job['id']].directory/'results/checkpoint.jsonl'
        self.assertEqual(json.loads(cfg2.read_text())['error_code'],'http_402')

    def test_three_failures_stop_remaining_tasks(self):
        class Broken(Model):
            def complete(self,*a,**kw): raise RuntimeError('do-not-serialize-this-secret')
        self.app.factory=Broken
        job=self.post('/api/jobs',self.config());state=self.wait(job['id'])
        self.assertEqual(state['status'],'stopped')
        self.assertEqual(state['completed'],3);self.assertEqual(state['failed'],3)
        self.assertIn('stop_reason',state)
        self.assertNotIn('do-not-serialize-this-secret',json.dumps(state))

    def test_single_generator_transport_and_no_evaluator_config(self):
        instances=[]
        class RecordingModel(Model):
            def __init__(self,**kwargs):
                instances.append(kwargs);super().__init__(**kwargs)
        self.app.factory=RecordingModel
        cfg=self.config();cfg.update(timeout=90,max_output_tokens=8192,thinking='disabled')
        job=self.post('/api/jobs',cfg);state=self.wait(job['id'])
        self.assertEqual(len(instances),1)
        self.assertEqual(instances[0]['thinking'],'disabled')
        self.assertEqual(instances[0]['timeout'],90)
        self.assertEqual(state['scored'],4)
        manifest=json.loads((self.app.jobs[job['id']].directory/'results/manifest.json').read_text())
        self.assertEqual(manifest['evaluation_method'],'exact_fact_id_matching_v1')
        self.assertFalse(any(k.startswith('judge_') for k in manifest))
        cfg=self.config();cfg['judge_model']='qwen3.8-max'
        with self.assertRaises(urllib.error.HTTPError) as error:self.post('/api/jobs',cfg)
        self.assertEqual(error.exception.code,400)

    def test_changed_upload_cannot_reuse_old_labels(self):
        from run_fact_experiments import jsonl,BENCHMARK
        case=jsonl(BENCHMARK)[0];case['query']='A different question'
        cfg=self.config();cfg['dataset_jsonl']=json.dumps(case)
        with self.assertRaises(urllib.error.HTTPError) as error:self.post('/api/jobs',cfg)
        self.assertEqual(error.exception.code,400)

    def test_stop_and_report_partial_results(self):
        entered, release=threading.Event(),threading.Event()
        class Delayed(Model):
            def complete(self,*a,**kw):
                entered.set();release.wait(5)
                return super().complete(*a,**kw)
        self.app.factory=Delayed
        job=self.post('/api/jobs',self.config())
        self.assertTrue(entered.wait(2))
        self.post('/api/jobs/'+job['id']+'/stop',{})
        release.set();state=self.wait(job['id'])
        self.assertEqual(state['status'],'stopped');self.assertLess(state['completed'],state['total'])
        self.assertTrue(self.client.open(self.url+'/api/jobs/'+job['id']+'/download').read())

if __name__=='__main__':unittest.main()
