import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from evaluate_generation import rescore
from experiment_execution import execute_with_retries
from generation_experiment_design import select_cases
from rag_generator import GenerationError
from run_generation_experiments import BENCHMARK, jsonl, run
from test_rouge_generation import SummaryModel


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.cases=select_cases(jsonl(BENCHMARK))[:3]
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'run'

    def experiment(self,model,**kwargs):
        options=dict(approaches=['baseline'],concurrency=1,retry_backoff=0,as_of='2026-11-07T00:00:00Z')
        options.update(kwargs)
        return run(self.cases[:1],self.out,client=model,model='scripted',**options)

    def test_bad_json_then_bad_citation_then_valid_output_retries_same_prompt_and_counts_usage(self):
        class Recovering(SummaryModel):
            def complete(self,*args,**kwargs):
                response=super().complete(*args,**kwargs)
                if len(self.calls)==1:response['content']='not JSON'
                elif len(self.calls)==2:response['content']=json.dumps({'claims':[{'text':'Wrong ID','chunk_ids':['wrong']} ]})
                return response
        model=Recovering()
        row=self.experiment(model,input_price=1,output_price=2)[0]
        self.assertEqual(row['status'],'completed')
        self.assertEqual(row['attempt_count'],3);self.assertEqual(row['retry_count'],2)
        self.assertEqual([a['status'] for a in row['attempts']],['failed','failed','completed'])
        self.assertEqual([a['error_code'] for a in row['attempts'][:2]],['invalid_json','invalid_citations'])
        self.assertEqual(model.calls,[model.calls[0]]*3)
        self.assertEqual(row['metrics']['model_calls'],3)
        self.assertEqual(row['metrics']['prompt_tokens'],60)
        self.assertEqual(row['metrics']['completion_tokens'],30)
        self.assertAlmostEqual(row['metrics']['estimated_cost_usd'],.00012)
        self.assertEqual(row['result']['usage']['prompt_tokens'],20)
        self.assertGreaterEqual(row['metrics']['latency_seconds'],sum(a['latency_seconds'] for a in row['attempts']))
        self.assertEqual(rescore([row]),[row])

    def test_exhaustion_records_one_row_per_task_and_all_failed_attempt_cost(self):
        class Broken(SummaryModel):
            def complete(self,*args,**kwargs):
                response=super().complete(*args,**kwargs);response['content']='{}';return response
        model=Broken();rows=self.experiment(model,approaches=['baseline','few_shot'],concurrency=2,
                                            input_price=1,output_price=2)
        self.assertEqual(len(rows),2);self.assertEqual(len(model.calls),6)
        self.assertTrue(all(r['status']=='failed' and r['attempt_count']==3 for r in rows))
        checkpoint=[json.loads(l) for l in (self.out/'checkpoint.jsonl').read_text().splitlines()]
        self.assertEqual(checkpoint,rows)
        report=json.loads((self.out/'report_data.json').read_text())
        self.assertEqual(report['request_totals']['model_calls'],6)
        self.assertEqual(report['request_totals']['prompt_tokens'],120)
        self.assertAlmostEqual(report['request_totals']['estimated_cost_usd'],.00024)
        self.assertTrue(all(r['total_retries']==2 for r in report['aggregates']))

    def test_valid_abstention_is_not_retried(self):
        class Abstaining(SummaryModel):
            def complete(self,*args,**kwargs):
                response=super().complete(*args,**kwargs);response['content']='{"claims":[]}';return response
        model=Abstaining();row=self.experiment(model)[0]
        self.assertEqual(row['status'],'completed');self.assertEqual(row['result']['status'],'abstained')
        self.assertEqual(row['attempt_count'],1);self.assertEqual(len(model.calls),1)

    def test_transient_errors_retry_and_unknown_usage_is_not_zero(self):
        class Flaky(SummaryModel):
            count=0
            def complete(self,*args,**kwargs):
                self.count+=1
                if self.count==1:raise GenerationError('private',code='http_5xx',http_status=503)
                if self.count==2:raise GenerationError('private',code='timeout')
                return super().complete(*args,**kwargs)
        model=Flaky();row=self.experiment(model)[0]
        self.assertEqual((row['status'],row['attempt_count']),('completed',3))
        self.assertIsNone(row['metrics']['prompt_tokens'])
        self.assertEqual(row['request_totals']['observed_prompt_tokens'],20)
        self.assertEqual(row['metrics']['model_calls'],3)
        self.assertNotIn('private',json.dumps(row))

    def test_permanent_account_error_stops_dispatch_and_is_not_retried(self):
        class Unauthorized(SummaryModel):
            count=0
            def complete(self,*a,**kw):
                self.count+=1;raise GenerationError('secret',code='http_401',http_status=401)
        model=Unauthorized();rows=self.experiment(model,approaches=['baseline','few_shot','compressed','combined'])
        self.assertEqual((len(rows),model.count),(1,1))
        summary=json.loads((self.out/'run_summary.json').read_text())
        self.assertTrue(summary['stopped']);self.assertEqual(summary['planned_tasks'],4)

    def test_real_parallelism_is_bounded_and_checkpoint_callbacks_are_serialized(self):
        class Parallel(SummaryModel):
            def __init__(self):
                super().__init__();self.lock=threading.Lock();self.ready=threading.Event();self.active=0;self.peak=0
            def complete(self,*a,**kw):
                with self.lock:
                    self.active+=1;self.peak=max(self.peak,self.active)
                    if self.active==3:self.ready.set()
                try:
                    if not self.ready.wait(2):raise AssertionError('Requests did not overlap')
                    time.sleep(.01)
                    return super().complete(*a,**kw)
                finally:
                    with self.lock:self.active-=1
        model=Parallel();threads=[]
        rows=run(self.cases,self.out,client=model,model='scripted',concurrency=3,
            progress=lambda *a:threads.append(threading.get_ident()))
        self.assertEqual(model.peak,3);self.assertEqual(model.active,0)
        self.assertEqual(len(rows),12);self.assertTrue(all(r['status']=='completed' for r in rows))
        self.assertEqual(len({(r['case_id'],r['approach'],r['repeat']) for r in rows}),12)
        self.assertEqual(set(threads),{threading.get_ident()})
        self.assertEqual(len((self.out/'checkpoint.jsonl').read_text().splitlines()),12)

    def test_stop_during_retry_wait_does_not_start_another_request(self):
        stop=threading.Event()
        class Limited(SummaryModel):
            count=0
            def complete(self,*a,**kw):
                self.count+=1;raise GenerationError('limited',code='http_429',retry_after_seconds=30)
        model=Limited()
        row=self.experiment(model,stop_requested=stop.is_set,
            activity=lambda e:stop.set() if e['type']=='retry_wait' else None)[0]
        self.assertEqual(model.count,1);self.assertTrue(row['retry_interrupted'])

    def test_backoff_honors_retry_after_and_second_attempt_delay(self):
        class Model:
            count=0
            def complete(self,*a,**kw):
                self.count+=1
                if self.count==1:raise GenerationError('limited',code='http_429',retry_after_seconds=7)
                if self.count==2:raise GenerationError('bad',code='invalid_json')
                return {'content':'{}','usage':{'prompt_tokens':1,'completion_tokens':1}}
        model=Model()
        with patch('experiment_execution.wait_for_retry',return_value=True) as waiter:
            row=execute_with_retries(lambda c:c.complete([],model='scripted',temperature=0),model)
        self.assertEqual([call.args[0] for call in waiter.call_args_list],[7,2])
        self.assertEqual(row['attempt_count'],3)

    def test_retry_disable_and_execution_config_validation(self):
        class Bad(SummaryModel):
            def complete(self,*args,**kwargs):
                response=super().complete(*args,**kwargs);response['content']='{}';return response
        model=Bad();row=self.experiment(model,max_retries=0)[0]
        self.assertEqual(row['attempt_count'],1)
        for option in ({'concurrency':0},{'concurrency':501},{'concurrency':True},
                       {'max_retries':3},{'max_retries':True},{'retry_backoff':-1}):
            with self.assertRaises(ValueError):self.experiment(model,**option)


if __name__=='__main__':unittest.main()
