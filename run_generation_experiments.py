"""Run A--D summarization experiments and score ROUGE against reviewed references."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from evaluate_generation import (EVALUATION_METHOD, F1_METRICS, REFERENCES, SCORING_CONFIG,
                                 jsonl, load_references, score_summary)
from rag_generator import APPROACHES, PROMPTS, OpenAICompatibleClient, build_context, generate_summary, safe_error
from review_context import context_time

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT/'experiments/steam_generation_eval.jsonl'
DRAFTS = ROOT/'data/reference_summaries/drafts/reference_drafts.jsonl'
BENCHMARK_CONTEXT_POLICY = 'full_review'


def dump(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2), encoding='utf-8')
    temp.replace(path)


def benchmark(cases):
    refs = load_references()
    original = {c['id']:c for c in jsonl(BENCHMARK)}
    drafts = {r['id']:r for r in jsonl(DRAFTS)}
    if not isinstance(cases,list) or not cases:
        raise ValueError('Choose at least one fixed benchmark question')
    seen = set()
    for case in cases:
        if not isinstance(case,dict):
            raise ValueError('Each case must be a benchmark record')
        cid = case.get('id')
        if not isinstance(cid, str):
            raise ValueError('Case ID must be a string')
        expected = original.get(cid)
        if (cid in seen or expected is None or cid not in refs
                or case.get('query') != expected['query'] or case.get('results') != expected['results']):
            raise ValueError('Questions, review text, order and metadata must match the approved fixed benchmark')
        seen.add(cid)
        draft = drafts[cid]
        if any(refs[cid][k] != draft[k] for k in ('context_sha256','review_metadata_sha256')):
            raise ValueError('Reference question/evidence metadata revision mismatch')
        context = build_context(case['results'], context_policy=BENCHMARK_CONTEXT_POLICY)
        if context != draft['fixed_context']:
            raise ValueError('Reference text context differs from the benchmark context')
    return refs


class SharedDraft:
    def __init__(self,client,cache,key):
        self.client,self.cache,self.key,self.replayed = client,cache,key,None

    def complete(self,messages,model,temperature):
        draft = messages[0]['content'].startswith((PROMPTS/'improved.txt').read_text())
        if draft and self.key in self.cache:
            self.replayed = self.cache[self.key]
            return self.replayed['response']
        start = time.perf_counter()
        response = self.client.complete(messages,model,temperature)
        if draft:
            self.cache[self.key] = {'response':response,'latency_seconds':time.perf_counter()-start}
        return response


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['approach']].append(row)
    table = []
    for arm,group in groups.items():
        valid = [r for r in group if r['status']=='completed']
        result = {'approach':arm,'attempted':len(group),'completed':len(valid),'failed':len(group)-len(valid)}
        score_fields = {f'{metric}_{part}' for metric in ('rouge1','rouge2','rougeL') for part in ('precision','recall','f1')}
        for key in sorted(score_fields | {k for r in valid for k in r['metrics']}):
            values = [r['metrics'][key] for r in valid if r['metrics'].get(key) is not None]
            result[key] = statistics.fmean(values) if values else None
            result[key+'_n'] = len(values)
        latency = sorted(r['metrics']['latency_seconds'] for r in valid if r['metrics'].get('latency_seconds') is not None)
        for p in (.5,.95):
            result['p'+str(int(p*100))+'_latency_seconds'] = latency[round((len(latency)-1)*p)] if latency else None
        for key in ('prompt_tokens','completion_tokens','estimated_cost_usd'):
            values = [r['metrics'].get(key) for r in valid]
            result['total_'+key] = sum(values) if values and all(v is not None for v in values) else None
        table.append(result)
    return sorted(table,key=lambda r:APPROACHES.index(r['approach']))


def paired_comparisons(rows, bootstrap_samples=5000, seed=42):
    by = {(r['case_id'],r['approach'],r.get('repeat',0)):r for r in rows if r['status']=='completed'}
    comparisons = []
    for before,after in [('baseline','improved'),('improved','verify_delete'),('improved','verify_revise')]:
        for metric in F1_METRICS:
            deltas = collections.defaultdict(list)
            for (qid,arm,repeat),a in by.items():
                b = by.get((qid,after,repeat))
                if arm==before and b is not None and metric in a['metrics'] and metric in b['metrics']:
                    deltas[qid].append(b['metrics'][metric]-a['metrics'][metric])
            values = [statistics.fmean(ds) for ds in deltas.values()]
            if not values:
                continue
            rng = random.Random(seed)
            boots = sorted(statistics.fmean(rng.choices(values,k=len(values))) for _ in range(bootstrap_samples))
            comparisons.append({'before':before,'after':after,'metric':metric,'n':len(values),
                'paired_rows':sum(len(ds) for ds in deltas.values()),'mean_delta':statistics.fmean(values),
                'bootstrap_95_percent_CI':[boots[int(.025*bootstrap_samples)],boots[int(.975*bootstrap_samples)-1]],
                'wins':sum(v>1e-12 for v in values),'ties':sum(abs(v)<=1e-12 for v in values),
                'losses':sum(v< -1e-12 for v in values),'bootstrap_samples':bootstrap_samples,'seed':seed})
    return comparisons


def export(out,rows):
    table = aggregate(rows); paired = paired_comparisons(rows)
    dump(out/'generation_results.json',rows)
    dump(out/'report_data.json',{'evaluation_method':EVALUATION_METHOD,'scoring':SCORING_CONFIG,
        'reference_sha256':hashlib.sha256(REFERENCES.read_bytes()).hexdigest(),
        'aggregates':table,'paired_comparisons':paired})
    fields = sorted({k for r in table for k in r})
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(table)
    predictions = [{'id':r['case_id'],'approach':r['approach'],'repeat':r.get('repeat',0),
                    'summary':r['result']['summary']} for r in rows if r['status']=='completed']
    (out/'predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions),encoding='utf-8')
    lines = ['# ROUGE summarization results','',
        'ROUGE compares generated English summaries with human-reviewed reference answers.',
        'Citation ID markers are removed; text is lowercased into English alphanumeric tokens without stemming.',
        'ROUGE-L uses whole-answer LCS. Scores in JSON/CSV are 0–1; this table shows F1 percentages.',
        'All nine precision/recall/F1 metrics and their valid counts are in summary.csv.',
        'No evaluation model is called. C/D self-checking remains part of generation.','',
        '| Method | Completed | Failed | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Latency s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        values = [f"{100*r[k]:.2f}" if r.get(k) is not None else 'N/A' for k in F1_METRICS]
        lines.append('| '+' | '.join([r['approach'],str(r['completed']),str(r['failed']),*values,str(r.get('latency_seconds','N/A'))])+' |')
    lines += ['','## Paired ROUGE comparisons','',
              '| After - before | Metric | Queries | Delta (pp) | Bootstrap 95% CI (pp) |',
              '|---|---|---:|---:|---:|']
    for p in paired:
        lo,hi = p['bootstrap_95_percent_CI']
        lines.append(f"| {p['after']} - {p['before']} | {p['metric']} | {p['n']} | {100*p['mean_delta']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] |")
    lines += ['','Bootstrap sampling uses questions as units, keeping repetitions together.',
              'B/C/D share an initial draft; reported logical usage includes one draft per method.',
              'Reference answers, evidence IDs, prompt hashes and settings are included in the export.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return table


def run(cases,out,*,client,model,approaches=APPROACHES,repeats=1,max_claims=4,seed=42,
        input_price=None,output_price=None,config=None,progress=None,stop_requested=None,as_of=None):
    refs = benchmark(cases)
    if (type(repeats) is not int or not 1<=repeats<=5 or type(max_claims) is not int or not 1<=max_claims<=12
            or not approaches or len(set(approaches))!=len(approaches) or any(a not in APPROACHES for a in approaches)):
        raise ValueError('Invalid experiment size or methods')
    for price in (input_price,output_price):
        if price is not None and (not math.isfinite(price) or price<0):
            raise ValueError('Invalid token price')
    clock = context_time(as_of).isoformat()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    manifest = dict(config or {},evaluation_method=EVALUATION_METHOD,scoring=SCORING_CONFIG,
        reference_version=refs[cases[0]['id']]['dataset_revision'],model=model,
        context_policy=BENCHMARK_CONTEXT_POLICY,include_review_metadata=True,as_of_utc=clock,
        approaches=list(approaches),repeats=repeats,max_claims=max_claims,seed=seed,
        reference_summaries_in_prompt=False,fact_candidates_in_prompt=False,
        context_limits={'max_context_chars':24000,'max_chunk_chars':4000},
        prompt_hashes={p:hashlib.sha256((PROMPTS/p).read_bytes()).hexdigest() for p in ('baseline.txt','improved.txt','verify.txt','revise.txt')},
        data_hash=hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('rag_generator.py','review_context.py','run_generation_experiments.py','evaluate_generation.py')},
        reference_sha256=hashlib.sha256(REFERENCES.read_bytes()).hexdigest())
    dump(out/'manifest.json',manifest)
    for name,items in [('input.jsonl',cases),('evaluation_references.jsonl',[refs[c['id']] for c in cases])]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    tasks = [(c,n,a) for c in cases for n in range(repeats) for a in approaches]
    random.Random(seed).shuffle(tasks)
    cache,rows = {},[]
    for case,repeat,arm in tasks:
        if stop_requested and stop_requested():
            break
        row = {'case_id':case['id'],'repeat':repeat,'approach':arm,'status':'completed',
               'evaluation_method':EVALUATION_METHOD,'reference_id':case['id'],
               'reference':refs[case['id']]['reference'],'metrics':{}}
        try:
            transport = SharedDraft(client,cache,(case['id'],repeat))
            result = generate_summary(case['query'],case['results'],approach=arm,model=model,client=transport,
                max_claims=max_claims,input_cost_per_million=input_price,output_cost_per_million=output_price,
                context_policy=BENCHMARK_CONTEXT_POLICY,include_review_metadata=True,as_of=clock)
            expected = build_context(case['results'],context_policy=BENCHMARK_CONTEXT_POLICY)
            actual = [{k:e[k] for k in ('chunk_id','text','game_name','source_url')} for e in result['context_evidence']]
            if actual != expected:
                raise ValueError('Model evidence differs from the approved reference context')
            row['measured_execution_seconds'] = result['latency_seconds']
            if transport.replayed:
                first = result['stage_calls'][0]
                result['latency_seconds'] += transport.replayed['latency_seconds']-first['latency_seconds']
                first.update(latency_seconds=transport.replayed['latency_seconds'],shared_draft=True)
            row['result'] = result
            metrics = score_summary(result['summary'],row['reference'],result['context_chunk_ids'])
            metrics.update(latency_seconds=result['latency_seconds'],model_calls=len(result['stage_calls']),
                prompt_tokens=result['usage'].get('prompt_tokens'),completion_tokens=result['usage'].get('completion_tokens'),
                estimated_cost_usd=result['estimated_cost_usd'],final_claims=len(result['claims']),
                provenance_rejected=result['validation']['rejected_claims'])
            row['metrics'] = metrics
        except Exception as error:
            row.update(status='failed',**safe_error(error))
        with (out/'checkpoint.jsonl').open('a',encoding='utf-8') as file:
            file.write(json.dumps(row,ensure_ascii=False)+'\n')
        rows.append(row)
        if progress:
            progress(row,len(rows),len(tasks))
    export(out,rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--model',default='deepseek-v4-pro');parser.add_argument('--base-url',default='https://api.deepseek.com')
    parser.add_argument('--limit',type=int,default=3);parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--max-claims',type=int,default=4);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--approaches',nargs='+',choices=APPROACHES,default=APPROACHES)
    parser.add_argument('--timeout',type=int,default=180);parser.add_argument('--max-output-tokens',type=int,default=8192)
    parser.add_argument('--thinking',choices=('auto','disabled','enabled'),default='auto')
    parser.add_argument('--input-price',type=float);parser.add_argument('--output-price',type=float)
    parser.add_argument('--as-of',help='Fixed ISO UTC reference time for reproducible review-age metadata')
    args = parser.parse_args()
    if not 1<=args.limit<=100:
        parser.error('limit must be between 1 and 100')
    client = OpenAICompatibleClient(base_url=args.base_url,timeout=args.timeout,max_tokens=args.max_output_tokens,thinking=args.thinking)
    if not client.api_key:
        parser.error('Set GENERATION_API_KEY or OPENAI_API_KEY in your own terminal')
    run(jsonl(BENCHMARK)[:args.limit],args.output,client=client,model=args.model,approaches=args.approaches,
        repeats=args.repeats,max_claims=args.max_claims,seed=args.seed,input_price=args.input_price,
        output_price=args.output_price,as_of=args.as_of,
        config={k:getattr(args,k) for k in ('base_url','timeout','max_output_tokens','thinking')})


if __name__=='__main__':
    main()
