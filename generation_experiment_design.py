"""Frozen data split and configurations for the few-shot x compression ablation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESIGN_DIR = ROOT/'data/generation_experiment'
SPLIT_PATH = DESIGN_DIR/'split.json'
EXAMPLES_PATH = DESIGN_DIR/'few_shot_examples.json'
EXPERIMENT_PROTOCOL = 'fewshot_compression_2x2_v1'
APPROACHES = ('baseline','few_shot','compressed','combined')
METHODS = {
    'baseline': {'few_shot':False,'compression':False,'label':'Baseline'},
    'few_shot': {'few_shot':True,'compression':False,'label':'Few-shot'},
    'compressed': {'few_shot':False,'compression':True,'label':'Compression'},
    'combined': {'few_shot':True,'compression':True,'label':'Few-shot + compression'},
}


def load_split():
    data = json.loads(SPLIT_PATH.read_text(encoding='utf-8'))
    dev, test, demos = data['development_ids'], data['test_ids'], data['example_ids']
    if (len(dev)!=20 or len(test)!=80 or len(set(dev+test))!=100
            or len(demos)!=2 or len(set(demos))!=2 or not set(demos)<=set(dev)):
        raise ValueError('Invalid frozen development/test/example split')
    return data


def select_cases(cases, split='test'):
    if split not in ('test','development'):
        raise ValueError('Choose test or development split')
    data = load_split()
    # The two demonstrations are never scored in either partition.
    ids = set(data['test_ids'] if split=='test' else data['development_ids']) - set(data['example_ids'])
    return [c for c in cases if c['id'] in ids]


def validate_cases(cases, split):
    if len(select_cases(cases,split))!=len(cases):
        raise ValueError('Questions must belong to the selected split; demonstration questions cannot be scored')


def load_examples(cases, references):
    from rag_generator import build_context, validate_claim
    data = json.loads(EXAMPLES_PATH.read_text(encoding='utf-8'))
    split = load_split()
    if [e.get('case_id') for e in data] != split['example_ids']:
        raise ValueError('Examples must be the two frozen development questions')
    canonical = {c['id']:c for c in cases}
    for e in data:
        case = canonical[e['case_id']]
        sources = {s['chunk_id']:s for s in build_context(case['results'],context_policy='full_review')}
        evidence = e['evidence']
        if (e['question']!=case['query'] or not evidence
                or any(sources.get(s.get('chunk_id'))!=s for s in evidence)):
            raise ValueError('Demonstration evidence differs from original reviews')
        visible = {s['chunk_id']:s for s in evidence}
        claims = e['response']['claims']
        if (not claims or any(validate_claim(c,visible) is None for c in claims)
                or ' '.join(c['text'] for c in claims)!=references[e['case_id']]['reference']):
            raise ValueError('Demonstrations must match reviewed development answers and valid source IDs')
    return data


def design_hashes():
    return {'split_sha256':hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
            'examples_sha256':hashlib.sha256(EXAMPLES_PATH.read_bytes()).hexdigest()}
