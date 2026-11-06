import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from annotate_generation_facts import ASPECTS
from rag_generator import generate_summary
from run_fact_experiments import BENCHMARK,ANNOTATIONS,jsonl,benchmark,catalog,run
from evaluate_fact_selection import score_selection


class FactModel:
    def __init__(self):self.calls=[];self.unknown=False;self.delete=False
    def complete(self,messages,model,temperature):
        self.calls.append(messages);payload=json.loads(messages[1]['content'])
        if 'evidence verifier' in messages[0]['content']:
            obj={'verdict':'unsupported' if self.delete else 'supported','reason':'scripted test'}
        else:
            h=payload['evidence'][0]
            obj={'claims':[{'text':'A reviewer discusses an aspect.','chunk_ids':[h['chunk_id']],
                'fact_ids':['F_UNKNOWN' if self.unknown else payload['fact_candidates'][0]['fact_id']],
                'evidence':[{'chunk_id':h['chunk_id'],'quote':h['text'][:30]}]}]}
        return {'content':json.dumps(obj),'usage':{'prompt_tokens':20,'completion_tokens':10}}


class FactExperimentTests(unittest.TestCase):
    def setUp(self):self.case=jsonl(BENCHMARK)[0]
    def test_catalog_does_not_reveal_aspect_presence_or_labels(self):
        from rag_generator import build_context
        refs,samples=benchmark([self.case]);context=build_context(self.case['results'])
        candidates=catalog(context,samples)
        self.assertEqual(len(candidates),len(context)*8)
        for h in context:
            self.assertEqual({c['aspect'] for c in candidates if c['chunk_id']==h['chunk_id']},set(ASPECTS))
        self.assertTrue(all(set(c)=={'fact_id','chunk_id','aspect'} for c in candidates))
        model=FactModel()
        generate_summary(self.case['query'],self.case['results'],approach='improved',client=model,fact_candidates=candidates)
        payload=json.loads(model.calls[0][1]['content'])
        for forbidden in ('reference_fact_ids','context_fact_ids','coverage_units','contrast_groups','human_annotation','target_sentiment'):
            self.assertNotIn(forbidden,payload)
        self.assertIn('ALL EIGHT',model.calls[0][0]['content'])
    def test_objective_scores_are_reproducible_and_drafts_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=FactModel()
            with patch('evaluate_generation.judge_result',side_effect=AssertionError('Evaluator must not run')):
                rows=run([self.case],Path(tmp)/'results',client=model,model='scripted')
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['status']=='completed' for r in rows))
            self.assertEqual(len(model.calls),4) # two drafts and two self-verifications
            ref=next(r for r in jsonl(ANNOTATIONS/'query_annotations.jsonl') if r['id']==self.case['id'])
            for r in rows:
                expected=score_selection(ref,{'selected_fact_ids':r['result']['selected_fact_ids']})
                for k,v in expected.items():self.assertEqual(r['metrics'][k],v)
            self.assertTrue((Path(tmp)/'results/predictions.jsonl').exists())
    def test_wrong_fact_ids_are_not_filtered_into_better_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=FactModel();model.unknown=True
            rows=run([self.case],Path(tmp)/'results',client=model,model='scripted',approaches=['baseline'])
            self.assertEqual(rows[0]['result']['selected_fact_ids'],['F_UNKNOWN'])
            self.assertEqual(rows[0]['metrics']['fact_selection_precision'],0)
            self.assertEqual(rows[0]['metrics']['unknown_candidate_id_count'],1)
    def test_deleting_a_claim_deletes_its_fact_selection(self):
        from rag_generator import build_context
        _,samples=benchmark([self.case]);candidates=catalog(build_context(self.case['results']),samples)
        model=FactModel();model.delete=True
        result=generate_summary(self.case['query'],self.case['results'],approach='verify_delete',client=model,fact_candidates=candidates)
        self.assertEqual(result['selected_fact_ids'],[])
        self.assertEqual(result['claims'],[])
    def test_missing_fact_ids_is_a_visible_schema_failure(self):
        class Missing(FactModel):
            def complete(self,*a,**kw):
                response=super().complete(*a,**kw)
                data=json.loads(response['content'])
                for c in data.get('claims',[]):c.pop('fact_ids',None)
                response['content']=json.dumps(data);return response
        with tempfile.TemporaryDirectory() as tmp:
            rows=run([self.case],Path(tmp)/'results',client=Missing(),model='scripted',approaches=['baseline'])
            self.assertEqual(rows[0]['status'],'failed')
            self.assertEqual(rows[0]['error_code'],'invalid_fact_ids')

    def test_changed_evidence_or_query_is_rejected(self):
        case=copy.deepcopy(self.case);case['results'][0]['text']='Different text'
        with self.assertRaises(ValueError):benchmark([case])
        case=copy.deepcopy(self.case);case['query']='Different query'
        with self.assertRaises(ValueError):benchmark([case])

if __name__=='__main__':unittest.main()
