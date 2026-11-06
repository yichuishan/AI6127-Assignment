import json
import unittest

from evaluate_fact_selection import ANNOTATIONS, evaluate, read_jsonl, score_selection


class FactAnnotationTests(unittest.TestCase):
    def target(self):
        return {'id':'q','reference_fact_ids':['F1','F2','F3'],
                'context_fact_ids':['F1','F2','F3','BG','EX'],
                'optional_fact_ids':['BG'],'explicit_excluded_fact_ids':['EX'],
                'coverage_units':[{'supporting_fact_ids':['F1','F2']},{'supporting_fact_ids':['F3']}],
                'contrast_groups':[{'required_for_query':True,'positive_fact_ids':['F1','F2'],
                                    'negative_fact_ids':['F3']}]}

    def test_duplicates_and_opinion_groups(self):
        m=score_selection(self.target(),{'selected_fact_ids':['F1','F3','F3']})
        self.assertEqual(m['coverage_unit_recall'],1)
        self.assertEqual(m['polarity_contrast_coverage'],1)
        self.assertAlmostEqual(m['fact_selection_recall'],2/3)
        self.assertEqual(m['duplicate_fact_id_count'],1)

    def test_optional_is_neither_required_nor_rewarded(self):
        m=score_selection(self.target(),{'selected_fact_ids':['F1','BG','EX']})
        self.assertEqual(m['fact_selection_precision'],.5)
        self.assertEqual(m['fact_selection_recall'],1/3)
        self.assertEqual(m['optional_selected_fact_count'],1)
        self.assertEqual(m['raw_selected_fact_count'],3)
        self.assertEqual(m['selected_fact_count'],2)
        self.assertEqual(m['explicit_excluded_selected_fact_count'],1)
        self.assertEqual(m['scored_false_positive_count'],1)
        self.assertEqual(m['polarity_contrast_coverage'],0)

    def test_unknown_ids_remain_errors(self):
        m=score_selection(self.target(),{'selected_fact_ids':['F1','unknown','BG']})
        self.assertEqual(m['fact_selection_precision'],.5)
        self.assertEqual(m['out_of_context_fact_rate'],1/3)
        self.assertIsNone(m['condition_retention_rate'])

    def test_empty_core_can_include_optional_background(self):
        target=self.target();target.update(reference_fact_ids=[],coverage_units=[],contrast_groups=[])
        m=score_selection(target,{'selected_fact_ids':['BG']})
        self.assertIsNone(m['fact_selection_f1'])
        self.assertIsNone(m['fact_selection_precision'])
        self.assertEqual(m['empty_core_reference_correct_abstention'],1)
        self.assertEqual(score_selection(target,{'selected_fact_ids':['EX']})['empty_core_reference_correct_abstention'],0)
        self.assertEqual(score_selection(self.target(),{'selected_fact_ids':[]})['fact_selection_f1'],0)

    def test_invalid_and_ambiguous_inputs(self):
        with self.assertRaises(ValueError):score_selection(self.target(),{'selected_fact_ids':None})
        target=self.target();target['optional_fact_ids']=['F1']
        with self.assertRaises(ValueError):score_selection(target,{'selected_fact_ids':[]})
        with self.assertRaises(ValueError):evaluate([self.target()],[{'id':'unknown','selected_fact_ids':[]}])
        prediction={'id':'q','selected_fact_ids':['F1']}
        with self.assertRaises(ValueError):evaluate([self.target()],[prediction,prediction])

    def test_final_dataset_and_user_decisions_are_consistent(self):
        refs=read_jsonl(ANNOTATIONS/'query_annotations.jsonl')
        facts=read_jsonl(ANNOTATIONS/'facts.jsonl')
        passages=read_jsonl(ANNOTATIONS/'passages.jsonl')
        review=json.loads((ANNOTATIONS/'human_review.json').read_text())
        self.assertEqual((len(passages),len(facts),len(refs),len(review['items'])),(1000,2898,100,40))
        self.assertEqual(sum(len(r['reference_fact_ids']) for r in refs),591)
        self.assertEqual(sum(len(r['coverage_units']) for r in refs),238)
        self.assertEqual(sum(d['required_for_query'] for r in refs for d in r['contrast_groups']),27)
        by={r['id']:r for r in refs}
        for item in review['items']:
            r=by[item['query_id']];fid=item['fact_id']
            field={'A':'reference_fact_ids','B':'optional_fact_ids','C':'explicit_excluded_fact_ids'}[item['answer']]
            self.assertIn(fid,r[field])
        for r in refs:
            core=set(r['reference_fact_ids'])
            self.assertFalse(core & set(r['optional_fact_ids']))
            self.assertFalse(core & set(r['explicit_excluded_fact_ids']))
            self.assertEqual(set().union(set(),*(set(u['supporting_fact_ids']) for u in r['coverage_units'])),core)
        self.assertIn('F_a113_graphics_visuals',by['hybrid_rerank/q69']['reference_fact_ids'])
        self.assertNotIn('F_a583_gameplay',by['hybrid_rerank/q12']['reference_fact_ids'])


if __name__=='__main__':unittest.main()
