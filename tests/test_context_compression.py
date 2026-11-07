import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from context_compression import GAP, compress_context
from generation_experiment_design import (load_examples,
    load_split, select_cases, validate_cases)
from evaluate_generation import load_references, rescore
from rag_generator import build_context
from run_generation_experiments import BENCHMARK, interaction_effects, jsonl, paired_comparisons, run
from test_rouge_generation import SummaryModel


def source(cid,text):
    return {'chunk_id':cid,'text':text,'game_name':'Test','source_url':'https://example.com/'+cid,
            'helpful_votes':10,'is_high_helpfulness':True}


class CompressionTests(unittest.TestCase):
    def test_whole_sentence_offsets_and_negation_neighbor_are_preserved(self):
        text='The graphics are bright. It does not crash on my old PC. However, large parks lag. The music is pleasant.'
        context=[source('s',text)]
        actual,info=compress_context('Does it crash on an old PC?',context,.8)
        self.assertIn('It does not crash on my old PC.',actual[0]['text'])
        self.assertIn('However, large parks lag.',actual[0]['text'])
        spans=info['retained_spans'][0]['spans']
        self.assertEqual(actual[0]['text'],GAP.join(text[a:b] for a,b in spans))
        self.assertEqual(actual[0]['source_url'],context[0]['source_url'])

    def test_deterministic_budget_no_input_mutation_and_identity(self):
        ctx=[source('a','The combat feels good. The music sounds nice. The combat is precise.'),
             source('b','The menu is slow. The story is long. The combat is repetitive.')]
        old=copy.deepcopy(ctx)
        a,info=compress_context('How is combat?',ctx,.5)
        self.assertEqual((a,info),compress_context('How is combat?',ctx,.5))
        self.assertEqual(ctx,old)
        self.assertLessEqual(info['retained_text_chars'],info['target_text_chars'])
        self.assertEqual(compress_context('combat',ctx,1)[0],ctx)
        self.assertEqual(compress_context('combat',[],.6)[0],[])

    def test_tiny_budget_falls_back_to_one_whole_sentence_and_records_overrun(self):
        text='This is one long indivisible statement about combat without any other sentence'
        result,info=compress_context('combat',[source('a',text)],.2)
        self.assertEqual(result[0]['text'],text)
        self.assertTrue(info['over_budget'])
        for invalid in (0,.1,1.1,True,float('nan'),'0.6'):
            with self.assertRaises(ValueError):compress_context('combat',[],invalid)

    def test_all_fixed_context_extracts_are_traceable_and_keep_ids_metadata_and_order(self):
        for case in jsonl(BENCHMARK):
            ctx=build_context(case['results'],context_policy='full_review',include_review_metadata=True,
                              as_of='2026-11-07T00:00:00Z')
            compressed,info=compress_context(case['query'],ctx)
            self.assertTrue(compressed)
            originals={e['chunk_id']:e for e in ctx}
            positions={e['chunk_id']:i for i,e in enumerate(ctx)}
            self.assertEqual([positions[e['chunk_id']] for e in compressed],sorted(positions[e['chunk_id']] for e in compressed))
            for e,retained in zip(compressed,info['retained_spans']):
                original=originals[e['chunk_id']]
                self.assertEqual(e['text'],GAP.join(original['text'][a:b] for a,b in retained['spans']))
                self.assertEqual({k:v for k,v in e.items() if k!='text'},
                                 {k:v for k,v in original.items() if k!='text'})


class DesignTests(unittest.TestCase):
    def test_frozen_stratified_split_and_development_demonstrations(self):
        cases=jsonl(BENCHMARK);split=load_split()
        self.assertEqual(set(split['development_ids'])|set(split['test_ids']),{c['id'] for c in cases})
        self.assertFalse(set(split['development_ids'])&set(split['test_ids']))
        self.assertEqual(Counter(c['query_type'] for c in cases if c['id'] in split['development_ids']),split['development_counts'])
        self.assertEqual(len(select_cases(cases)),80)
        self.assertEqual(len(select_cases(cases,'development')),18)
        demos=load_examples(cases,load_references())
        self.assertEqual([d['case_id'] for d in demos],split['example_ids'])
        for split_name in ('test','development'):
            self.assertFalse(set(split['example_ids'])&{c['id'] for c in select_cases(cases,split_name)})
        with self.assertRaises(ValueError):validate_cases([next(c for c in cases if c['id']==split['example_ids'][0])],'development')

    def test_no_old_self_check_arms_or_wrong_split_can_launch_a_model_call(self):
        cases=jsonl(BENCHMARK);model=SummaryModel()
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ('improved','verify_delete','verify_revise'):
                with self.assertRaises(ValueError):run([select_cases(cases)[0]],Path(tmp)/arm,client=model,model='scripted',approaches=[arm])
            with self.assertRaises(ValueError):run([select_cases(cases,'development')[0]],Path(tmp)/'wrong',client=model,model='scripted')
        self.assertEqual(model.calls,[])

    def test_four_arms_differ_only_by_examples_and_compression_and_recompute_checks_spans(self):
        model=SummaryModel();case=select_cases(jsonl(BENCHMARK))[0]
        with tempfile.TemporaryDirectory() as tmp:
            rows=run([case],Path(tmp)/'run',client=model,model='scripted',input_price=1,output_price=2,
                     compression_ratio=.5,as_of='2026-11-07T00:00:00Z')
            self.assertTrue(all(r['status']=='completed' for r in rows))
            self.assertEqual(len(model.calls),4)
            self.assertEqual(sum('examples' in json.loads(m[1]['content']) for m in model.calls),2)
            full=next(r['result']['context_evidence'] for r in rows if r['approach']=='baseline')
            for row in rows:
                result=row['result'];arm=row['approach']
                self.assertEqual(len(result['stage_calls']),1)
                self.assertEqual(result['approach'],arm)
                if arm in ('compressed','combined'):
                    self.assertEqual(result['context_evidence'],compress_context(case['query'],full,.5)[0])
                else:self.assertEqual(result['context_evidence'],full)
                self.assertEqual(row['metrics']['prompt_tokens'],20)
                self.assertIsNotNone(row['metrics']['estimated_cost_usd'])
            self.assertEqual(rescore(rows),rows)
            changed=copy.deepcopy(rows)
            next(r for r in changed if r['approach']=='compressed')['result']['compression']['retained_spans'][0]['spans'][0][0]+=1
            with self.assertRaises(ValueError):rescore(changed)
            changed=copy.deepcopy(rows)
            next(r for r in changed if r['approach']=='few_shot')['result']['few_shot_example_ids']=['test-leak']
            with self.assertRaises(ValueError):rescore(changed)

    def test_factorial_and_paired_resource_comparisons_use_matched_questions(self):
        rows=[{'case_id':'q1','approach':arm,'repeat':0,'status':'completed',
               'metrics':{'rougeL_f1':value,'prompt_tokens':tokens,'latency_seconds':1}}
              for arm,value,tokens in [('baseline',.2,100),('few_shot',.3,120),('compressed',.25,60),('combined',.4,80)]]
        effects=interaction_effects(rows,bootstrap_samples=100)
        self.assertAlmostEqual(effects[0]['mean_interaction'],.05)
        pairs=paired_comparisons(rows,bootstrap_samples=100)
        reduction=next(p for p in pairs if p['after']=='compressed' and p['before']=='baseline' and p['metric']=='prompt_tokens')
        self.assertEqual(reduction['mean_delta'],-40)
        self.assertEqual(reduction['bootstrap_95_percent_CI'],[-40,-40])


if __name__=='__main__':unittest.main()
