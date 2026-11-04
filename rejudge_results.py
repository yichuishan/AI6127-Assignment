"""Re-evaluate saved summaries; never call the generator or alter the source ZIP."""
import argparse
import copy
import getpass
import hashlib
import json
import os
from pathlib import Path
import zipfile

from evaluate_generation import judge_result
from rag_generator import OpenAICompatibleClient, build_context, safe_error
from run_generation_experiments import export_report, save


def load_archive(path):
    with zipfile.ZipFile(path) as archive:
        required = ('input.jsonl','results/checkpoint.jsonl','results/manifest.json')
        data = {}
        for name in required:
            if archive.namelist().count(name) != 1 or archive.getinfo(name).file_size > 32_000_000:
                raise ValueError('Missing, duplicated or oversized experiment file: '+name)
            data[name] = archive.read(name).decode('utf-8')
    cases = [json.loads(x) for x in data['input.jsonl'].splitlines() if x.strip()]
    rows = [json.loads(x) for x in data['results/checkpoint.jsonl'].splitlines() if x.strip()]
    if not cases or not rows or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Invalid or empty archived experiment')
    keys = [(r['case_id'],r['repeat'],r['approach']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate archived task IDs')
    return cases, rows, json.loads(data['results/manifest.json'])


def rejudge(path, output, *, client=None, model=None, base_url=None, thinking='auto',
            timeout=300, max_tokens=4096, all_judges=False, input_price=None, output_price=None):
    cases, original, manifest = load_archive(path)
    cases = {c['id']:c for c in cases}
    model = model or manifest.get('judge_model')
    base_url = base_url or manifest.get('effective_judge_base_url') or manifest.get('judge_base_url') or manifest.get('effective_base_url')
    if not model or not base_url:
        raise ValueError('Provide a judge model and base URL')
    out = Path(output)
    if out.exists():
        raise ValueError('Use a new output directory; refusing to overwrite previous results')
    transport = client or OpenAICompatibleClient(base_url=base_url,timeout=timeout,max_tokens=max_tokens,thinking=thinking)
    rows = copy.deepcopy(original)
    selected = [r for r in rows if r.get('result') and (all_judges or r['status']=='judge_failed' or r.get('error_stage')=='judge')]
    out.mkdir(parents=True)
    input_price = manifest.get('judge_input_price') if input_price is None else input_price
    output_price = manifest.get('judge_output_price') if output_price is None else output_price
    save(out/'rejudge_manifest.json',{'source_zip_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'source_manifest':manifest,'judge_model':model,'judge_base_url':base_url,'judge_thinking':thinking,
        'judge_timeout':timeout,'judge_max_output_tokens':max_tokens,'all_judges':all_judges,
        'selected_tasks':len(selected),'judge_input_price':input_price,'judge_output_price':output_price,
        'protocol':'explicit_lengths_v2','note':'No generation calls. Untouched successful judges retain their old protocol; use --all-judges for a consistent final evaluation.'})
    (out/'input.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases.values()),encoding='utf-8')
    (out/'checkpoint.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    print('Saved generation outputs:',len(rows),'selected judge calls:',len(selected),flush=True)
    for i,row in enumerate(selected,1):
        case = cases[row['case_id']]
        previous = {k:row[k] for k in ('status','judge','judge_attempt','error_type','error_code','error_stage','error_message','validation_issues') if k in row}
        row.setdefault('rejudge_history',[]).append(previous)
        for key in ('judge','judge_attempt','error_type','error_code','error_stage','error_message','http_status','validation_issues'):
            row.pop(key,None)
        # Remove only final judge metrics; preserve all generation and reference metrics.
        for key in ('faithfulness','relevance','claim_faithfulness_fraction','citation_precision',
                    'judge_context_relevance_fraction','key_point_coverage','judge_prompt_tokens',
                    'judge_completion_tokens','estimated_judge_cost_usd'):
            row['metrics'].pop(key,None)
        context = build_context(case['results'],manifest.get('max_context_chars',24000),manifest.get('max_chunk_chars',4000))
        try:
            judge = judge_result(case['query'],row['result'],case['results'],transport,model,
                case.get('key_points'),context=context,detailed=True)
            row.update(status='completed',generation_status='completed',judge_status='completed',judge=judge)
            scores, usage, metrics = judge['scores'],judge['usage'],row['metrics']
            for key in ('faithfulness','relevance'):
                metrics[key] = scores[key]['score']
            for source,target in [('claim_support','claim_faithfulness_fraction'),('citation_support','citation_precision'),('context_relevance','judge_context_relevance_fraction')]:
                values = scores[source]; metrics[target] = sum(values)/len(values) if values else None
            if scores.get('coverage'):
                metrics['key_point_coverage'] = sum(x['covered'] for x in scores['coverage'])/len(scores['coverage'])
            metrics.update(judge_prompt_tokens=usage.get('prompt_tokens'),judge_completion_tokens=usage.get('completion_tokens'),estimated_judge_cost_usd=None)
            if input_price is not None and output_price is not None and all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
                metrics['estimated_judge_cost_usd']=(usage['prompt_tokens']*input_price+usage['completion_tokens']*output_price)/1e6
        except Exception as error:
            row.update(status='judge_failed',generation_status='completed',judge_status='failed',**safe_error(error))
            if hasattr(error,'judge_attempt'):
                row['judge_attempt'] = error.judge_attempt
        # Snapshot all rows after every attempt, so interruption retains finished work.
        temp = out/'checkpoint.jsonl.tmp'
        temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        temp.replace(out/'checkpoint.jsonl')
        print(i,'/',len(selected),row['case_id'],row['approach'],row['status'],flush=True)
    (out/'input.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases.values()),encoding='utf-8')
    export_report(out,rows,manifest.get('seed',42))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--output')
    parser.add_argument('--model')
    parser.add_argument('--base-url')
    parser.add_argument('--thinking',choices=['auto','disabled','enabled'],default='auto')
    parser.add_argument('--timeout',type=int,default=300)
    parser.add_argument('--max-tokens',type=int,default=4096)
    parser.add_argument('--all-judges',action='store_true')
    parser.add_argument('--api-key-file')
    args = parser.parse_args()
    _,_,config = load_archive(args.archive)
    model = args.model or config.get('judge_model')
    base_url = args.base_url or config.get('effective_judge_base_url') or config.get('judge_base_url') or config.get('effective_base_url')
    if not model or not base_url:
        parser.error('Supply --model and --base-url')
    print('Judge:',model,'API:',base_url)
    key = Path(args.api_key_file).read_text().strip() if args.api_key_file else os.getenv('JUDGE_API_KEY')
    if not key:
        key = getpass.getpass('Judge API Key (hidden): ').strip()
    if not key:
        parser.error('An evaluator API key is required')
    client = OpenAICompatibleClient(base_url=base_url,api_key=key,thinking=args.thinking,timeout=args.timeout,max_tokens=args.max_tokens)
    rejudge(args.archive,args.output or str(Path(args.archive).with_suffix('.rejudged')),
        client=client,model=model,base_url=base_url,thinking=args.thinking,timeout=args.timeout,
        max_tokens=args.max_tokens,all_judges=args.all_judges)

if __name__=='__main__':main()
