"""Offline ROUGE rescoring by default; --legacy-facts reproduces archived fact-ID scores."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import zipfile

from evaluate_fact_selection import ANNOTATIONS, REFERENCE_VERSION, read_jsonl, score_selection
from rag_generator import APPROACHES
from run_fact_experiments import BENCHMARK, benchmark, dump, export


def load_results(path):
    path=Path(path)
    if path.is_dir():
        candidates=[path/'generation_results.json',path/'results/generation_results.json']
        path=next((p for p in candidates if p.exists()),candidates[0])
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as archive:
            # Canonical evidence must match before archived outputs use these references.
            cases=[json.loads(x) for x in archive.read('input.jsonl').splitlines() if x.strip()]
            benchmark(cases)
            if 'results/checkpoint.jsonl' in archive.namelist():
                return [json.loads(x) for x in archive.read('results/checkpoint.jsonl').splitlines() if x.strip()]
            return json.loads(archive.read('results/generation_results.json'))
    if path.suffix.lower()=='.jsonl':
        return read_jsonl(path)
    return json.loads(path.read_text(encoding='utf-8'))


def rescore(rows, references):
    if not isinstance(rows,list) or not rows:
        raise ValueError('Expected nonempty experiment result rows')
    targets={r['id']:r for r in references}
    if len(targets)!=len(references):
        raise ValueError('Duplicate reference IDs')
    out=copy.deepcopy(rows)
    seen=set()
    execution_fields=('latency_seconds','model_calls','prompt_tokens','completion_tokens',
                      'estimated_cost_usd','final_claims','unknown_candidate_id_count','provenance_rejected')
    for row in out:
        key=(row['case_id'],row['approach'],row.get('repeat',0))
        if key in seen or key[0] not in targets or key[1] not in APPROACHES:
            raise ValueError('Unknown or duplicate result case/approach/repeat')
        seen.add(key)
        if row['status']=='failed':
            continue
        if row['status']!='completed':
            raise ValueError('Expected final fact-selection experiment status')
        ref=targets[row['case_id']]
        if row['result'].get('context_policy','full_review')!='full_review':
            raise ValueError('Latest-update outputs require references for their filtered context, not the full-review benchmark')
        if row['result'].get('context_chunk_ids')!=ref['context_chunk_ids']:
            raise ValueError('Result context differs from the fixed benchmark')
        previous=row['metrics']
        metrics=score_selection(ref,{'selected_fact_ids':row['result']['selected_fact_ids']})
        metrics.update({k:previous.get(k) for k in execution_fields})
        row['metrics']=metrics
    return out


def fact_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',help='Downloaded experiment ZIP, results directory, or generation_results.json')
    parser.add_argument('--output',required=True,help='New directory for recomputed outputs')
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory')
    references=read_jsonl(ANNOTATIONS/'query_annotations.jsonl')
    rows=rescore(load_results(args.source),references)
    output.mkdir(parents=True,exist_ok=True)
    table=export(output,rows)
    used={r['case_id'] for r in rows}
    (output/'evaluation_references.jsonl').write_text(''.join(
        json.dumps(r,ensure_ascii=False)+'\n' for r in references if r['id'] in used),encoding='utf-8')
    (output/'input.jsonl').write_text(''.join(
        json.dumps(c,ensure_ascii=False)+'\n' for c in read_jsonl(BENCHMARK) if c['id'] in used),encoding='utf-8')
    dump(output/'rescore_manifest.json',{'reference_version':REFERENCE_VERSION,'uses_model':False,
        'source':Path(args.source).name,
        'reference_sha256':hashlib.sha256((ANNOTATIONS/'query_annotations.jsonl').read_bytes()).hexdigest(),
        'human_review_sha256':hashlib.sha256((ANNOTATIONS/'human_review.json').read_bytes()).hexdigest(),
        'scorer_sha256':hashlib.sha256(Path(__file__).with_name('evaluate_fact_selection.py').read_bytes()).hexdigest()})
    for item in table:
        print(f"{item['approach']}: F1={item.get('fact_selection_f1')} completed={item['completed']}")


def main():
    import sys
    if '--legacy-facts' in sys.argv:
        sys.argv.remove('--legacy-facts')
        fact_main()
    else:
        from evaluate_generation import main as rouge_main
        rouge_main()


if __name__=='__main__':main()
