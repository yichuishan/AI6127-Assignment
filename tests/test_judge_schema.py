import copy
import json
import unittest
from evaluate_generation import judge_result
from rag_generator import GenerationError, safe_error


class JudgeSchemaTests(unittest.TestCase):
    def setUp(self):
        self.hits=[{'chunk_id':'r1','text':'One review.'},{'chunk_id':'r2','text':'Another review.'}]
        self.result={'summary':'Summary','claims':[{'text':'Claim','chunk_ids':['r1','r2'],'evidence':[]}]}
        self.valid={'faithfulness':{'score':4,'reason':'Grounded'},'relevance':{'score':3,'reason':'Relevant'},
            'claim_support':[True],'citation_support':[True,False],'context_relevance':[True,False]}
    def call(self,obj):
        class Client:
            calls=[]
            def complete(self,messages,model,temperature):
                self.calls.append(messages)
                return {'content':json.dumps(obj),'usage':{'prompt_tokens':10,'completion_tokens':4}}
        client=Client();result=judge_result('query',self.result,self.hits,client,'mock',detailed=True)
        return result,client
    def test_prompt_specifies_exact_lengths_and_citation_pairs(self):
        result,client=self.call(self.valid)
        payload=json.loads(client.calls[-1][1]['content'])
        self.assertEqual(payload['required_array_lengths'],{'claim_support':1,'citation_support':2,'context_relevance':2})
        self.assertEqual(len(payload['citation_pairs']),2)
        self.assertEqual(len(payload['output_template']['citation_support']),2)
        self.assertEqual(result['protocol'],'explicit_lengths_v2')
    def test_representation_normalization_is_audited(self):
        obj=copy.deepcopy(self.valid);obj['faithfulness']['score']='4';obj['relevance']['score']=3.0
        obj['claim_support']=['true'];obj['citation_support']=[1,0]
        result,_=self.call(obj)
        self.assertEqual(result['scores']['citation_support'],[True,False])
        self.assertEqual(result['scores']['faithfulness']['score'],4)
        self.assertEqual(len(result['normalizations']),5)
    def test_missing_or_wrong_length_is_not_padded(self):
        obj=copy.deepcopy(self.valid);obj['citation_support']=[True]
        with self.assertRaises(GenerationError) as error:self.call(obj)
        details=safe_error(error.exception)
        issue=details['validation_issues'][0]
        self.assertEqual(issue,{'field':'citation_support','problem':'wrong_length','expected_length':2,'actual_length':1})
        self.assertEqual(error.exception.judge_attempt['usage']['completion_tokens'],4)
    def test_ambiguous_scores_and_nested_arrays_are_rejected(self):
        for obj in [dict(self.valid,claim_support=[{'supported':True}]),
                    dict(self.valid,citation_support=[[True,False]]),
                    dict(self.valid,faithfulness={'score':True,'reason':'Not a numeric score'})]:
            with self.assertRaises(GenerationError):self.call(obj)

if __name__=='__main__':unittest.main()
