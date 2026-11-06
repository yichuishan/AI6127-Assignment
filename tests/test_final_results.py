import copy
import json
from pathlib import Path
import statistics
import unittest
from unittest.mock import patch

from evaluate_fact_selection import ANNOTATIONS, paired_comparisons, read_jsonl
from recompute_results import load_results, rescore


ROOT=Path(__file__).resolve().parents[1]


class FinalResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=load_results(ROOT/'experiments/final/generation_results.json')
        cls.refs=read_jsonl(ANNOTATIONS/'query_annotations.jsonl')

    def test_all_saved_scores_and_report_numbers_reproduce_without_api(self):
        with patch('rag_generator.OpenAICompatibleClient.complete',side_effect=AssertionError('No API allowed')):
            recalculated=rescore(self.rows,self.refs)
        self.assertEqual(len(recalculated),400)
        self.assertEqual(sum(r['status']=='completed' for r in recalculated),398)
        for old,new in zip(self.rows,recalculated):
            self.assertEqual(old['metrics'],new['metrics'])
        expected={
            'baseline':(.8540357142857142,.7271233816688362,.7581800018684342,.8873496873496873,.5625),
            'improved':(.880297619047619,.6968075134741801,.7512300445174073,.8717652717652717,.5),
            'verify_delete':(.8948615160349854,.6867166280431587,.7473789992713351,.8704567541302235,.5),
            'verify_revise':(.8919703595724003,.6957545289177942,.7523277919752299,.8704567541302235,.5)}
        metrics=['fact_selection_precision','fact_selection_recall','fact_selection_f1',
                 'coverage_unit_recall','polarity_contrast_coverage']
        for arm,values in expected.items():
            own=[r for r in recalculated if r['approach']==arm and r['status']=='completed']
            for metric,value in zip(metrics,values):
                observed=statistics.mean(r['metrics'][metric] for r in own if r['metrics'][metric] is not None)
                self.assertAlmostEqual(observed,value,places=12)

    def test_optional_only_answer_and_mixed_substatement(self):
        rows=rescore(self.rows,self.refs)
        q76=next(r for r in rows if r['case_id']=='hybrid_rerank/q76' and r['approach']=='verify_revise')
        self.assertEqual(q76['metrics']['optional_selected_fact_count'],1)
        self.assertIsNone(q76['metrics']['fact_selection_f1'])
        q69=next(r for r in rows if r['case_id']=='hybrid_rerank/q69' and r['approach']=='improved')
        self.assertAlmostEqual(q69['metrics']['fact_selection_f1'],4/7)

    def test_changed_context_and_duplicate_predictions_are_rejected(self):
        row=copy.deepcopy(next(r for r in self.rows if r['status']=='completed'))
        with self.assertRaises(ValueError):rescore([row,row],self.refs)
        row['result']['context_chunk_ids']=['changed']
        with self.assertRaises(ValueError):rescore([row],self.refs)

    def test_paired_report_comparisons_use_matching_queries(self):
        pairs=paired_comparisons(self.rows,bootstrap_samples=200)
        f1=[p for p in pairs if p['metric']=='fact_selection_f1']
        self.assertEqual([p['n'] for p in f1],[99,98,98])
        for p,expected in zip(f1,[-.006949957351026869,-.0013125763125763118,.0025024295432458706]):
            self.assertAlmostEqual(p['mean_delta'],expected,places=12)

    def test_bootstrap_keeps_repeats_in_the_same_query(self):
        def row(arm,repeat,value):
            return {'case_id':'q','approach':arm,'repeat':repeat,'status':'completed',
                    'metrics':{'fact_selection_f1':value}}
        pairs=paired_comparisons([row('baseline',0,.2),row('improved',0,.4),
                                 row('baseline',1,.8),row('improved',1,.6)],bootstrap_samples=100)
        p=next(p for p in pairs if p['metric']=='fact_selection_f1')
        self.assertEqual((p['n'],p['paired_rows']),(1,2))
        self.assertAlmostEqual(p['mean_delta'],0)


if __name__=='__main__':unittest.main()
