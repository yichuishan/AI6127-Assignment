import copy
from pathlib import Path
import tempfile
import unittest
from annotate_generation_facts import build, match_quote, read_jsonl
from evaluate_fact_selection import evaluate, score_selection


class FactAnnotationTests(unittest.TestCase):
    def source(self):
        labels={a:{'sentiment':'Not mentioned','subjectivity':'','evidence':''} for a in
                ('story','graphics_visuals','gameplay','controls','performance','audio','bugs_stability','online_multiplayer')}
        records=[]
        for i,(text,sentiment) in enumerate([('On my RTX 3060, it runs smoothly after the update.','Positive'),
                                            ('It runs poorly in multiplayer.','Negative')],1):
            own=copy.deepcopy(labels);own['performance']={'sentiment':sentiment,'subjectivity':'Opinionated','evidence':text}
            records.append({'sample_id':'a00'+str(i),'chunk_id':'r'+str(i),'document_id':'d'+str(i),'review_id':str(i),
                'game_name':'Game','appid':1,'text':text,'status':'Confirmed','human_annotation':own})
        return {'records':records}
    def test_offsets_and_whitespace_mapping(self):
        result=match_quote('x runs\n  smoothly y','runs smoothly')
        self.assertEqual(result['status'],'whitespace_aligned')
        self.assertEqual(result['quote'],'runs\n  smoothly')
        self.assertEqual(match_quote('runs badly','runs smoothly')['status'],'unmatched')
    def test_full_build_and_label_provenance(self):
        source=self.source();query={'query_id':'q','query_text':'Performance?','target_games':['Game'],
            'target_aspects':['performance'],'target_sentiment':'any'}
        case={'id':'hybrid/q','query_id':'q','query':'Performance?','query_type':'single_aspect','retriever':'hybrid',
              'results':[{'chunk_id':r['chunk_id'],'text':r['text']} for r in source['records']]}
        with tempfile.TemporaryDirectory() as tmp:
            stats=build(source,[query],[case],tmp)
            facts=read_jsonl(Path(tmp)/'facts.jsonl');target=read_jsonl(Path(tmp)/'query_annotations.jsonl')[0]
            self.assertEqual(stats['facts'],2);self.assertEqual(stats['required_contrast_groups'],1)
            self.assertFalse(target['human_keypoint_reviewed'])
            conditions=read_jsonl(Path(tmp)/'conditions_candidates.jsonl')
            for c in conditions:
                r=next(r for r in source['records'] if r['chunk_id']==c['chunk_id'])
                self.assertEqual(r['text'][c['start']:c['end']],c['mention'])
                self.assertFalse(c['human_reviewed'])
            perfect=score_selection(target,{'selected_fact_ids':[f['fact_id'] for f in facts]})
            self.assertEqual(perfect['fact_selection_f1'],1)
            self.assertEqual(perfect['polarity_contrast_coverage'],1)
            self.assertIsNone(perfect['condition_retention_rate'])
            wrong=score_selection(target,{'selected_fact_ids':[facts[0]['fact_id'],'unknown']})
            self.assertEqual(wrong['fact_selection_precision'],.5)
            self.assertEqual(wrong['out_of_context_fact_rate'],.5)
            self.assertEqual(wrong['polarity_contrast_coverage'],0)
            with self.assertRaises(ValueError):build(source,[query],[case],tmp)
    def test_duplicate_reviews_do_not_inflate_unit_coverage(self):
        target={'reference_fact_ids':['F1','F2','F3'],'context_fact_ids':['F1','F2','F3'],
            'coverage_units':[{'supporting_fact_ids':['F1','F2']},{'supporting_fact_ids':['F3']}],
            'contrast_groups':[]}
        score=score_selection(target,{'selected_fact_ids':['F1','F3','F3']})
        self.assertEqual(score['coverage_unit_recall'],1)
        self.assertAlmostEqual(score['fact_selection_recall'],2/3)
        self.assertEqual(score['duplicate_fact_id_count'],1)
    def test_empty_reference_and_missing_predictions(self):
        target={'id':'q','reference_fact_ids':[],'context_fact_ids':['F1'],'coverage_units':[],'contrast_groups':[]}
        score=score_selection(target,{'selected_fact_ids':[]})
        self.assertIsNone(score['fact_selection_f1']);self.assertEqual(score['empty_reference_correct_abstention'],1)
        with self.assertRaises(ValueError):score_selection(target,{'selected_fact_ids':None})
        with self.assertRaises(ValueError):evaluate([target],[{'id':'unknown','selected_fact_ids':[]}])

if __name__=='__main__':unittest.main()
