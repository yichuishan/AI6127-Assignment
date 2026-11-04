import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from run_generation_experiments import agreement, context_metrics, parser, run, summarize

class ScriptedClient:
    def __init__(self): self.calls=[]
    def complete(self,messages,model,temperature):
        self.calls.append(messages)
        system=messages[0]['content']
        if 'Evaluate an answer' in system:
            value={'faithfulness':{'score':5,'reason':'scripted'},'relevance':{'score':4,'reason':'scripted'},
                'coverage':[{'covered':True,'reason':'scripted'}], 'claim_support':[True],
                'citation_support':[True],'context_relevance':[True]}
        elif 'evidence verifier' in system:
            value={'verdict':'supported','reason':'scripted'}
        else:
            value={'claims':[{'text':'One reviewer enjoys the story.','chunk_ids':['r1'],
                'evidence':[{'chunk_id':'r1','quote':'I enjoy'}]}]}
        return {'content':json.dumps(value),'usage':{'prompt_tokens':10,'completion_tokens':5}}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.case={'id':'q1','query':'How is the story?','results':[{'chunk_id':'r1','text':'I enjoy the story.'}],
                   'key_points':['A reviewer enjoys the story.'],'relevance_grades':{'r1':2}}
        self.dataset=self.root/'eval.jsonl';self.dataset.write_text(json.dumps(self.case)+'\n')
    def args(self,*extra):
        return parser().parse_args(['run',str(self.dataset),'--output',str(self.root/'out'),*extra])
    def test_paired_drafts_resume_metrics_and_no_credentials_in_artifacts(self):
        client=ScriptedClient();judge=ScriptedClient()
        args=self.args('--judge-model','scripted-judge','--input-price','1','--output-price','2',
                       '--judge-input-price','1','--judge-output-price','2')
        rows=run(args,client,judge)
        self.assertEqual(len(rows),4);self.assertTrue(all(r['status']=='completed' for r in rows))
        # baseline + one shared draft + two verification calls
        self.assertEqual(len(client.calls),4)
        self.assertEqual(len(judge.calls),4)
        self.assertTrue(all(r['metrics']['citation_precision']==1 for r in rows))
        self.assertTrue(all(r['metrics']['estimated_judge_cost_usd']>0 for r in rows))
        self.assertTrue(all(r['metrics']['estimated_cost_usd']>0 for r in rows))
        run(args,client,judge)
        self.assertEqual(len(client.calls),4)
        args.model='changed'
        with self.assertRaises(ValueError):run(args,client,judge)
    def test_key_file_value_and_path_are_not_saved(self):
        keyfile=self.root/'private-key'
        keyfile.write_text('test-private-secret-123')
        from unittest.mock import patch
        with patch.dict('os.environ',{},clear=False):
            run(self.args('--approaches','baseline','--api-key-file',str(keyfile)),ScriptedClient())
        for path in (self.root/'out').iterdir():
            content=path.read_text()
            self.assertNotIn('test-private-secret-123',content)
            self.assertNotIn(str(keyfile),content)

    def test_preflight_does_not_call_client(self):
        client=ScriptedClient();run(self.args('--preflight'),client)
        self.assertFalse(client.calls)
        self.assertTrue((self.root/'out/preflight.json').exists())
    def test_context_missing_labels_not_counted_as_irrelevant(self):
        metrics=context_metrics({'relevance_grades':{'r1':1}},[{'chunk_id':'r1'},{'chunk_id':'r2'}])
        self.assertEqual(metrics['context_judgment_coverage'],.5)
        self.assertEqual(metrics['context_relevance_fraction'],1)
    def test_failed_api_is_recorded_without_secret_or_fake_score(self):
        class Broken:
            def complete(self,*a,**k):raise RuntimeError('secret-do-not-save')
        rows=run(self.args('--approaches','baseline'),Broken())
        self.assertEqual(rows[0]['status'],'failed')
        self.assertNotIn('secret-do-not-save',(self.root/'out/checkpoint.jsonl').read_text())
        self.assertEqual(rows[0]['metrics'],{})
    def test_agreement_and_query_cluster_bootstrap(self):
        path=self.root/'labels.csv'
        path.write_text('record_id,annotator,label\na,A,pos\na,B,pos\nb,A,pos\nb,B,neg\n')
        args=parser().parse_args(['agreement',str(path),'--output',str(self.root/'agreement.json')]);agreement(args)
        obj=json.loads((self.root/'agreement.json').read_text());self.assertEqual(obj['pairs'][0]['agreement'],.5)
        rows=[{'case_id':'q','repeat':n,'retriever':'x','approach':arm,'status':'completed',
               'metrics':{'latency_seconds':1,'faithfulness':score}} for n in range(3)
              for arm,score in [('baseline',2),('improved',4)]]
        result=summarize(rows)['paired_query_bootstrap']
        faith=[x for x in result if x['metric']=='faithfulness'][0]
        self.assertEqual(faith['queries'],1);self.assertEqual(faith['mean_delta'],2)
    def test_exact_context_is_passed_to_judge(self):
        client=ScriptedClient();judge=ScriptedClient()
        rows=run(self.args('--approaches','improved','--judge-model','scripted','--max-chunk-chars','17'),client,judge)
        payload=json.loads(judge.calls[0][1]['content'])
        self.assertEqual(len(payload['evidence'][0]['text']),17)
        self.assertEqual(rows[0]['status'],'completed')

if __name__=='__main__':unittest.main()
