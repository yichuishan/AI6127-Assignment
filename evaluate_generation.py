"""ROUGE-1/2/L against reviewed English references. Standard library; no API calls."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parent
REFERENCES = ROOT / 'data/reference_summaries/reviewed/approved_references.jsonl'
EVALUATION_METHOD = 'rouge_1_2_l_human_references_v1'
F1_METRICS = ('rouge1_f1', 'rouge2_f1', 'rougeL_f1')
SCORING_CONFIG = {
    'metrics': ['rouge1', 'rouge2', 'rougeL'], 'stemming': False,
    'tokenizer': 'lowercase ASCII a-z/0-9 tokens (rouge-score default tokenizer)',
    'citations': 'remove original source-ID markers before tokenization',
    'aggregation': 'macro mean per completed answer; failed requests are separate',
    'scale': '0 to 1', 'rougeL_variant': 'whole-answer LCS, not rougeLsum',
}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def load_references(path=REFERENCES):
    records = jsonl(path)
    refs = {}
    for row in records:
        if (not isinstance(row.get('id'), str) or row['id'] in refs
                or row.get('human_reviewed') is not True or row.get('reference_language') != 'en'
                or not isinstance(row.get('reference'), str) or not row['reference'].strip()
                or not row.get('context_sha256') or not row.get('review_metadata_sha256')):
            raise ValueError('References must have unique IDs, approved English text and evidence hashes')
        refs[row['id']] = row
    if not refs:
        raise ValueError('No approved English reference summaries')
    return refs


def strip_citations(text, citation_ids=()):
    if not isinstance(text, str):
        raise ValueError('Summary and reference must be strings')
    # Unknown Steam-style citation markers are also formatting, not answer words.
    text = re.sub(r'\[steam:[^\]\s]+\]', '', text)
    for cid in set(citation_ids):
        if isinstance(cid, str):
            text = text.replace('[' + cid + ']', '')
    return text


def tokenize(text):
    # Matches google-research/rouge-score's use_stemmer=False tokenization.
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).split()


def prf(overlap, predicted, expected):
    precision = overlap / predicted if predicted else 0.0
    recall = overlap / expected if expected else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def lcs_length(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def score_summary(prediction, reference, citation_ids=()):
    predicted = tokenize(strip_citations(prediction, citation_ids))
    expected = tokenize(strip_citations(reference, citation_ids))
    metrics = {}
    for n in (1, 2):
        p = Counter(tuple(predicted[i:i+n]) for i in range(len(predicted)-n+1))
        r = Counter(tuple(expected[i:i+n]) for i in range(len(expected)-n+1))
        values = prf(sum((p & r).values()), sum(p.values()), sum(r.values()))
        metrics.update({f'rouge{n}_{name}': value for name, value in zip(('precision','recall','f1'), values)})
    values = prf(lcs_length(predicted, expected), len(predicted), len(expected))
    metrics.update({f'rougeL_{name}': value for name, value in zip(('precision','recall','f1'), values)})
    return metrics


def load_results(path):
    path = Path(path)
    if path.is_dir():
        path = next((p for p in (path/'generation_results.json', path/'results/generation_results.json') if p.exists()), path/'generation_results.json')
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if 'input.jsonl' in names:
                cases = [json.loads(line) for line in archive.read('input.jsonl').splitlines() if line.strip()]
                from run_generation_experiments import benchmark
                benchmark(cases)
            name = next((n for n in ('results/generation_results.json','generation_results.json',
                                    'results/checkpoint.jsonl','checkpoint.jsonl') if n in names), None)
            if name is None:
                raise ValueError('ZIP has no saved generation results')
            body = archive.read(name).decode('utf-8')
            return [json.loads(line) for line in body.splitlines() if line.strip()] if name.endswith('.jsonl') else json.loads(body)
    return jsonl(path) if path.suffix.lower() == '.jsonl' else json.loads(path.read_text(encoding='utf-8'))


def rescore(rows, references=None):
    """Score saved full/compressed answers; validate evidence and retain historical metrics."""
    from rag_generator import APPROACHES as LEGACY_APPROACHES, build_context
    from generation_experiment_design import (APPROACHES, EXPERIMENT_PROTOCOL, METHODS,
        load_split, validate_cases)
    from context_compression import compress_context
    refs = references if references is not None else load_references()
    cases = {c['id']: c for c in jsonl(ROOT/'experiments/steam_generation_eval.jsonl')}
    out, seen = copy.deepcopy(rows), set()
    if not isinstance(out, list) or not out:
        raise ValueError('Expected nonempty saved generation results')
    execution = ('latency_seconds','model_calls','prompt_tokens','completion_tokens',
                 'estimated_cost_usd','final_claims','provenance_rejected',
                 'context_preparation_seconds','original_context_text_chars','context_text_chars',
                 'context_text_ratio','context_chunks')
    for row in out:
        key = row['case_id'], row['approach'], row.get('repeat', 0)
        if key in seen or key[0] not in refs or key[1] not in set(APPROACHES)|set(LEGACY_APPROACHES):
            raise ValueError('Unknown or duplicate question/method/repeat')
        seen.add(key)
        if row['status'] == 'failed':
            continue
        if row['status'] != 'completed':
            raise ValueError('Expected completed/failed generation status')
        result = row['result']
        expected = build_context(cases[key[0]]['results'], context_policy='full_review')
        new_design=row.get('experiment_protocol')==EXPERIMENT_PROTOCOL
        if key[1] in ('few_shot','compressed','combined') and not new_design:
            raise ValueError('Missing experiment protocol for few-shot/compression results')
        if new_design:
            if key[1] not in APPROACHES:
                raise ValueError('Self-check arms are not part of the current experiment')
            validate_cases([cases[key[0]]],row.get('split'))
            expected=build_context(cases[key[0]]['results'],context_policy='full_review',
                include_review_metadata=True,as_of=result['context_as_of_utc'])
            settings=METHODS[key[1]]
            if settings['compression']:
                if result.get('original_context_evidence')!=expected:
                    raise ValueError('Original context differs from fixed evidence')
                expected,info=compress_context(cases[key[0]]['query'],expected,result['compression']['requested_ratio'])
                if info!=result['compression']:
                    raise ValueError('Compression parameters/spans differ from deterministic construction')
            if result.get('context_evidence')!=expected:
                raise ValueError('Actual evidence differs from deterministic context construction')
            example_ids=load_split()['example_ids'] if settings['few_shot'] else []
            if result.get('few_shot_example_ids',[])!=example_ids:
                raise ValueError('Saved demonstration IDs differ from frozen development examples')
        if (result.get('query') != cases[key[0]]['query']
                or result.get('context_policy','full_review') != 'full_review'
                or result.get('context_chunk_ids') != [e['chunk_id'] for e in expected]):
            raise ValueError('Saved answer context does not match the approved full-context references')
        if result.get('context_evidence') is not None and not new_design:
            visible = [{k:e.get(k) for k in ('chunk_id','text','game_name','source_url')} for e in result['context_evidence']]
            if visible != expected:
                raise ValueError('Saved answer evidence text differs from the approved reference context')
        old = row.get('metrics') or {}
        if old and any(not k.startswith('rouge') for k in old if k not in execution):
            row.setdefault('previous_metrics', copy.deepcopy(old))
        metrics = score_summary(result['summary'], refs[key[0]]['reference'], result['context_chunk_ids'])
        metrics.update({k:old.get(k) for k in execution})
        row.update(metrics=metrics, reference=refs[key[0]]['reference'],
                   reference_id=key[0], evaluation_method=EVALUATION_METHOD)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Results JSON, experiment ZIP, or directory')
    parser.add_argument('--output', required=True, help='A new directory for ROUGE reports')
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    from run_generation_experiments import benchmark, dump, export
    rows = rescore(load_results(args.source))
    selected = {row['case_id'] for row in rows}
    cases = [c for c in jsonl(ROOT/'experiments/steam_generation_eval.jsonl') if c['id'] in selected]
    refs = benchmark(cases)
    out.mkdir(parents=True, exist_ok=True)
    (out/'input.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases), encoding='utf-8')
    (out/'evaluation_references.jsonl').write_text(''.join(json.dumps(refs[c['id']],ensure_ascii=False)+'\n' for c in cases), encoding='utf-8')
    table = export(out, rows)
    dump(out/'rescore_manifest.json', {'evaluation_method':EVALUATION_METHOD,'uses_model':False,
         'source':Path(args.source).name,'reference_sha256':hashlib.sha256(REFERENCES.read_bytes()).hexdigest(),
         'scoring':SCORING_CONFIG,'scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for arm in table:
        print(f"{arm['approach']}: ROUGE-1={arm.get('rouge1_f1')} ROUGE-2={arm.get('rouge2_f1')} ROUGE-L={arm.get('rougeL_f1')}")


if __name__ == '__main__':
    main()
