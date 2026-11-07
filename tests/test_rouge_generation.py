import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from evaluate_generation import (EVALUATION_METHOD, load_references, load_results,
                                 rescore, score_summary, strip_citations, tokenize)
from run_generation_experiments import BENCHMARK, aggregate, benchmark, jsonl, paired_comparisons, run
from generation_experiment_design import EXAMPLES_PATH, EXPERIMENT_PROTOCOL, select_cases


class SummaryModel:
    def __init__(self):
        self.calls = []

    def complete(self,messages,model,temperature):
        self.calls.append(messages)
        payload = json.loads(messages[1]['content'])
        e = payload['evidence'][0]
        data = {'claims':[{'text':'The game offers an engaging story.','chunk_ids':[e['chunk_id']]}]}
        return {'content':json.dumps(data),'usage':{'prompt_tokens':20,'completion_tokens':10}}


class RougeScorerTests(unittest.TestCase):
    def test_identical_disjoint_and_empty_texts(self):
        scores = score_summary('Good story and smooth combat.', 'Good story and smooth combat.')
        self.assertTrue(all(value==1 for value in scores.values()))
        for pred,ref in [('a b','x y'),('','a b'),('a b',''),('','')]:
            self.assertTrue(all(value==0 for value in score_summary(pred,ref).values()))

    def test_precision_recall_orientation_and_multiset_counts(self):
        scores = score_summary('a a a b', 'a a b')
        self.assertAlmostEqual(scores['rouge1_precision'], .75)
        self.assertEqual(scores['rouge1_recall'], 1)
        self.assertAlmostEqual(scores['rouge1_f1'], 6/7)
        self.assertAlmostEqual(scores['rouge2_precision'], 2/3)
        self.assertEqual(scores['rouge2_recall'], 1)
        self.assertAlmostEqual(scores['rouge2_f1'], .8)

    def test_rouge_l_uses_order_and_rouge2_short_text_is_zero(self):
        scores = score_summary('a c b', 'a b c')
        self.assertEqual(scores['rouge1_f1'], 1)
        self.assertEqual(scores['rouge2_f1'], 0)
        self.assertAlmostEqual(scores['rougeL_f1'], 2/3)
        self.assertEqual(score_summary('good','good')['rouge2_f1'],0)

    def test_citations_are_removed_without_removing_other_bracketed_words(self):
        text = 'Good [steam:1:2:chunk:001] story [review_1] [RTX 4060].'
        cleaned = strip_citations(text,['review_1'])
        self.assertNotIn('steam:',cleaned)
        self.assertNotIn('review_1',cleaned)
        self.assertIn('[RTX 4060]',cleaned)
        self.assertEqual(tokenize(cleaned),['good','story','rtx','4060'])
        self.assertEqual(score_summary('GOOD story! [steam:1:2:chunk:001]','good story')['rougeL_f1'],1)
        self.assertNotEqual(score_summary('crashes','crash')['rouge1_f1'],1)  # no stemming

    def test_unreviewed_or_duplicate_references_are_rejected(self):
        row = next(iter(load_references().values()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'refs.jsonl'
            path.write_text(json.dumps(dict(row,human_reviewed=False))+'\n')
            with self.assertRaises(ValueError):load_references(path)
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError):load_references(path)


class RougeExperimentTests(unittest.TestCase):
    def setUp(self):
        self.case = select_cases(jsonl(BENCHMARK))[0]

    def test_four_independent_calls_only_development_examples_and_scores_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = SummaryModel();out=Path(tmp)/'run'
            rows = run([self.case],out,client=model,model='scripted',as_of='2026-11-07T00:00:00Z')
            self.assertEqual(len(rows),4)
            self.assertTrue(all(r['status']=='completed' for r in rows))
            self.assertEqual(len(model.calls),4)
            reference = load_references()[self.case['id']]['reference']
            for messages in model.calls:
                p = json.loads(messages[1]['content'])
                self.assertNotIn('fact_candidates',p)
                self.assertNotIn('reference',p)
                self.assertNotIn('evidence verifier',messages[0]['content'])
                if 'examples' in p:
                    self.assertEqual(list(p)[0],'examples')
                    self.assertEqual(list(p['examples']),['case1','case2'])
                    original=json.loads(EXAMPLES_PATH.read_text())
                    self.assertEqual(p['examples'],{'case1':original[0],'case2':original[1]})
                    self.assertNotIn(self.case['id'],[e['case_id'] for e in p['examples'].values()])
                    self.assertIn('Read case1 and case2 separately',messages[0]['content'])
                self.assertNotIn(reference,messages[1]['content'])
                self.assertNotIn('STRUCTURED FACT SELECTION',messages[0]['content'])
                self.assertIn('helpful_votes',p['evidence'][0])
                self.assertIn('is_high_helpfulness',p['evidence'][0])
            for row in rows:
                expected=score_summary(row['result']['summary'],reference,row['result']['context_chunk_ids'])
                self.assertNotIn('selected_fact_ids',row['result'])
                self.assertEqual(row['experiment_protocol'],EXPERIMENT_PROTOCOL)
                self.assertEqual(len(row['result']['stage_calls']),1)
                self.assertNotIn('semantic_check',row['result']['validation'])
                for k,v in expected.items():self.assertAlmostEqual(row['metrics'][k],v)
            report=json.loads((out/'report_data.json').read_text())
            self.assertEqual(report['evaluation_method'],EVALUATION_METHOD)
            self.assertIn('rougeL_f1',(out/'summary.csv').read_text())
            self.assertNotIn('fact_selection',(out/'summary.csv').read_text())
            self.assertNotIn('selected_fact_ids',(out/'predictions.jsonl').read_text())
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['experiment_protocol'],EXPERIMENT_PROTOCOL)
            self.assertEqual(manifest['split'],'test')
            self.assertEqual(manifest['scoring']['stemming'],False)
            self.assertEqual(rescore(rows),rows)

    def test_all_100_cases_have_matching_references_and_metadata_does_not_change_text(self):
        from rag_generator import build_context
        cases=jsonl(BENCHMARK);refs=benchmark(cases)
        self.assertEqual(len(refs),100)
        for case in cases:
            full=build_context(case['results'],context_policy='full_review')
            enhanced=build_context(case['results'],context_policy='full_review',include_review_metadata=True,
                                   as_of='2026-11-07T00:00:00Z')
            self.assertEqual([{k:e[k] for k in ('chunk_id','text','game_name','source_url')} for e in enhanced],full)
        solar=next(h for c in cases for h in c['results'] if '208994799' in h['chunk_id'])
        self.assertIn('60 FPS',build_context([solar],context_policy='full_review',include_review_metadata=True)[0]['text'])

    def test_changed_question_text_order_votes_and_duplicate_cases_are_rejected(self):
        for change in ('query','text','order','votes'):
            case=copy.deepcopy(self.case)
            if change=='query':case['query']='Different question'
            elif change=='text':case['results'][0]['text']='Different evidence'
            elif change=='order':case['results'].reverse()
            else:case['results'][0]['metadata']['helpful_votes']+=1
            with self.assertRaises(ValueError):benchmark([case])
        with self.assertRaises(ValueError):benchmark([self.case,self.case])

    def test_failures_stay_separate_and_zero_overlap_is_a_valid_score(self):
        rows=[{'approach':'baseline','status':'failed','metrics':{}},
              {'approach':'baseline','status':'completed','metrics':dict(score_summary('xyz','abc'),latency_seconds=1)}]
        agg=aggregate(rows)[0]
        self.assertEqual((agg['completed'],agg['failed']),(1,1))
        self.assertEqual(agg['rougeL_f1'],0)
        self.assertEqual(agg['rougeL_f1_n'],1)
        empty=aggregate(rows[:1])[0]
        self.assertIsNone(empty['rougeL_f1'])
        self.assertEqual(empty['rougeL_f1_n'],0)

    def test_bootstrap_groups_repetitions_by_question(self):
        def row(arm,repeat,value):
            return {'case_id':'q','approach':arm,'repeat':repeat,'status':'completed','metrics':{'rougeL_f1':value}}
        paired=paired_comparisons([row('baseline',0,.2),row('few_shot',0,.4),
                                  row('baseline',1,.8),row('few_shot',1,.6)],bootstrap_samples=100)
        self.assertEqual((paired[0]['n'],paired[0]['paired_rows']),(1,2))
        self.assertAlmostEqual(paired[0]['mean_delta'],0)

    def test_saved_results_and_zip_can_be_scored_without_api_and_reject_changed_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            rows=run([self.case],out,client=SummaryModel(),model='scripted',approaches=['baseline'])
            source=Path(tmp)/'run.zip'
            with zipfile.ZipFile(source,'w') as z:
                z.write(out/'input.jsonl','input.jsonl')
                z.write(out/'generation_results.json','results/generation_results.json')
            self.assertEqual(rescore(load_results(source)),rows)
            changed=copy.deepcopy(rows)
            changed[0]['result']['context_evidence'][0]['text']='Changed'
            with self.assertRaises(ValueError):rescore(changed)
            changed=copy.deepcopy(rows);changed[0]['result']['context_policy']='latest_update'
            with self.assertRaises(ValueError):rescore(changed)
            output=Path(tmp)/'rescored'
            process=subprocess.run([sys.executable,'evaluate_generation.py',str(source),'--output',str(output)],
                                   cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertTrue((output/'summary.csv').exists())


if __name__=='__main__':unittest.main()
