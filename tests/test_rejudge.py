import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from rejudge_results import load_archive,rejudge


class SavedJudge:
    def __init__(self,bad=False):self.calls=0;self.bad=bad
    def complete(self,messages,model,temperature):
        self.calls+=1
        assert messages[0]['content'].startswith('Evaluate an answer')
        obj=copy.deepcopy(json.loads(messages[1]['content'])['output_template'])
        obj['faithfulness']={'score':4,'reason':'Scripted support assessment'}
        obj['relevance']={'score':5,'reason':'Scripted relevance assessment'}
        if self.bad:obj['citation_support']=[]
        return {'content':json.dumps(obj),'usage':{'prompt_tokens':10,'completion_tokens':5}}


class RejudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.archive=self.root/'prior.zip'
        cases=[{'id':'q','query':'Question?','results':[{'chunk_id':'r','text':'Evidence.'}]}]
        result={'summary':'Saved summary [r]','claims':[{'text':'Saved summary','chunk_ids':['r'],'evidence':[{'chunk_id':'r','quote':'Evidence.'}]}]}
        self.rows=[{'case_id':'q','repeat':0,'approach':a,'retriever':'hybrid','status':status,
                    'result':copy.deepcopy(result),'metrics':{'latency_seconds':2.0,'prompt_tokens':100}}
                   for a,status in [('baseline','completed'),('improved','judge_failed')]]
        self.rows[1].update(error_stage='judge',error_code='invalid_judge')
        with zipfile.ZipFile(self.archive,'w') as z:
            z.writestr('input.jsonl',json.dumps(cases[0])+'\n')
            z.writestr('results/checkpoint.jsonl',''.join(json.dumps(r)+'\n' for r in self.rows))
            z.writestr('results/manifest.json',json.dumps({'judge_model':'qwen3.8-max','effective_judge_base_url':'https://example.com/v1'}))
    def test_only_failed_judges_are_called_and_summaries_preserved(self):
        before=hashlib.sha256(self.archive.read_bytes()).hexdigest();client=SavedJudge()
        rows=rejudge(self.archive,self.root/'out',client=client)
        self.assertEqual(client.calls,1)
        self.assertEqual(rows[0],self.rows[0])
        self.assertEqual(rows[1]['status'],'completed')
        self.assertEqual(rows[1]['result'],self.rows[1]['result'])
        self.assertEqual(rows[1]['metrics']['prompt_tokens'],100)
        self.assertEqual(hashlib.sha256(self.archive.read_bytes()).hexdigest(),before)
        self.assertTrue((self.root/'out/summary.csv').exists())
    def test_failed_rejudge_keeps_diagnostics_and_generation(self):
        rows=rejudge(self.archive,self.root/'out',client=SavedJudge(bad=True))
        self.assertEqual(rows[1]['status'],'judge_failed')
        self.assertEqual(rows[1]['validation_issues'][0]['field'],'citation_support')
        self.assertEqual(rows[1]['result'],self.rows[1]['result'])
    def test_all_judges_option_and_no_overwrite(self):
        client=SavedJudge();rejudge(self.archive,self.root/'out',client=client,all_judges=True)
        self.assertEqual(client.calls,2)
        with self.assertRaises(ValueError):rejudge(self.archive,self.root/'out',client=client)

if __name__=='__main__':unittest.main()
