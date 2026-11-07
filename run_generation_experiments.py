"""Run the few-shot x compression ablation with reviewed-reference ROUGE scoring."""
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
from rag_generator import PROMPTS, OpenAICompatibleClient, build_context, generate_summary, safe_error
from generation_experiment_design import (APPROACHES, METHODS, EXPERIMENT_PROTOCOL,
    SPLIT_PATH, EXAMPLES_PATH, design_hashes, load_split, load_examples, select_cases, validate_cases)
from context_compression import DEFAULT_RATIO, VERSION as COMPRESSION_VERSION, compress_context
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


def approach_order(arm):
    # Offline scoring of saved historical self-check runs remains supported.
    order = list(APPROACHES) + ['improved','verify_delete','verify_revise']
    return order.index(arm)


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
    return sorted(table,key=lambda r:approach_order(r['approach']))


def paired_comparisons(rows, bootstrap_samples=5000, seed=42):
    by = {(r['case_id'],r['approach'],r.get('repeat',0)):r for r in rows if r['status']=='completed'}
    comparisons = []
    new_design = any(r['approach'] in ('few_shot','compressed','combined') for r in rows)
    pairs = [('baseline','few_shot'),('baseline','compressed'),('baseline','combined'),
             ('few_shot','combined'),('compressed','combined')] if new_design else [
             ('baseline','improved'),('improved','verify_delete'),('improved','verify_revise')]
    metrics = (*F1_METRICS,'prompt_tokens','latency_seconds','estimated_cost_usd') if new_design else F1_METRICS
    for before,after in pairs:
        for metric in metrics:
            deltas = collections.defaultdict(list)
            for (qid,arm,repeat),a in by.items():
                b = by.get((qid,after,repeat))
                if arm==before and b is not None and a['metrics'].get(metric) is not None and b['metrics'].get(metric) is not None:
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


def interaction_effects(rows, bootstrap_samples=5000, seed=42):
    """Factorial interaction: combined - few-shot - compression + baseline."""
    by = {(r['case_id'],r['approach'],r.get('repeat',0)):r for r in rows if r['status']=='completed'}
    effects = []
    for metric in F1_METRICS:
        grouped = collections.defaultdict(list)
        for (qid,arm,repeat),base in by.items():
            if arm!='baseline':continue
            arms = [by.get((qid,a,repeat)) for a in APPROACHES]
            if any(r is None or r['metrics'].get(metric) is None for r in arms):continue
            a,b,c,d = [r['metrics'][metric] for r in arms]
            grouped[qid].append(d-b-c+a)
        values = [statistics.fmean(v) for v in grouped.values()]
        if not values:continue
        rng=random.Random(seed)
        boots=sorted(statistics.fmean(rng.choices(values,k=len(values))) for _ in range(bootstrap_samples))
        effects.append({'metric':metric,'n':len(values),'paired_rows':sum(map(len,grouped.values())),
            'mean_interaction':statistics.fmean(values),'bootstrap_95_percent_CI':[
                boots[int(.025*bootstrap_samples)],boots[int(.975*bootstrap_samples)-1]],
            'bootstrap_samples':bootstrap_samples,'seed':seed})
    return effects


def export(out,rows):
    table = aggregate(rows); paired = paired_comparisons(rows); effects=interaction_effects(rows)
    dump(out/'generation_results.json',rows)
    dump(out/'report_data.json',{'evaluation_method':EVALUATION_METHOD,'scoring':SCORING_CONFIG,
        'reference_sha256':hashlib.sha256(REFERENCES.read_bytes()).hexdigest(),
        'experiment_protocol':EXPERIMENT_PROTOCOL if any(r.get('experiment_protocol')==EXPERIMENT_PROTOCOL for r in rows) else 'historical',
        'aggregates':table,'paired_comparisons':paired,'factorial_interactions':effects})
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
        'Each current method uses one independent generation call; no self-check or evaluator is used.','',
        '| Method | Completed | Failed | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Input tokens | Latency s | Text retained |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        values = [f"{100*r[k]:.2f}" if r.get(k) is not None else 'N/A' for k in F1_METRICS]
        lines.append('| '+' | '.join([r['approach'],str(r['completed']),str(r['failed']),*values,
            str(r.get('prompt_tokens','N/A')),str(r.get('latency_seconds','N/A')),
            str(r.get('context_text_ratio','N/A'))])+' |')
    lines += ['','## Paired comparisons','',
              'ROUGE differences use percentage points; other metrics retain their original units.',
              '| After - before | Metric | Queries | Delta | Bootstrap 95% CI |',
              '|---|---|---:|---:|---:|']
    for p in paired:
        lo,hi = p['bootstrap_95_percent_CI']
        scale = 100 if p['metric'] in F1_METRICS else 1
        lines.append(f"| {p['after']} - {p['before']} | {p['metric']} | {p['n']} | {scale*p['mean_delta']:.6g} | [{scale*lo:.6g}, {scale*hi:.6g}] |")
    if effects:
        lines += ['','## Factorial interactions','',
            'Combined - few-shot - compression + baseline; ROUGE differences in percentage points.',
            '| Metric | Queries | Interaction (pp) | Bootstrap 95% CI (pp) |','|---|---:|---:|---:|']
        for p in effects:
            lo,hi=p['bootstrap_95_percent_CI']
            lines.append(f"| {p['metric']} | {p['n']} | {100*p['mean_interaction']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] |")
    lines += ['','Bootstrap sampling uses questions as units, keeping repetitions together.',
              'The split, demonstrations, full and actual evidence, compression spans and settings are saved for each new run.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return table


def run(cases,out,*,client,model,approaches=APPROACHES,repeats=1,max_claims=4,seed=42,
        input_price=None,output_price=None,config=None,progress=None,stop_requested=None,as_of=None,
        split='test',compression_ratio=DEFAULT_RATIO):
    refs = benchmark(cases)
    validate_cases(cases,split)
    if type(compression_ratio) not in (float,int) or not math.isfinite(compression_ratio) or not .2<=compression_ratio<=1:
        raise ValueError('Compression ratio must be between 0.2 and 1')
    all_cases=jsonl(BENCHMARK)
    examples=load_examples(all_cases,refs)
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
        experiment_protocol=EXPERIMENT_PROTOCOL,split=split,data_split=load_split(),
        design_hashes=design_hashes(),method_settings={a:METHODS[a] for a in approaches},
        compression={'version':COMPRESSION_VERSION,'ratio':compression_ratio},
        reference_version=refs[cases[0]['id']]['dataset_revision'],model=model,
        context_policy=BENCHMARK_CONTEXT_POLICY,include_review_metadata=True,as_of_utc=clock,
        approaches=list(approaches),repeats=repeats,max_claims=max_claims,seed=seed,
        current_reference_in_prompt=False,fact_candidates_in_prompt=False,
        development_demonstrations_in_few_shot_prompt=True,test_references_in_prompt=False,
        context_limits={'max_context_chars':24000,'max_chunk_chars':4000},
        prompt_hashes={p:hashlib.sha256((PROMPTS/p).read_bytes()).hexdigest() for p in ('baseline.txt','few_shot.txt')},
        data_hash=hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('rag_generator.py','review_context.py','run_generation_experiments.py','evaluate_generation.py','context_compression.py','generation_experiment_design.py')},
        reference_sha256=hashlib.sha256(REFERENCES.read_bytes()).hexdigest())
    dump(out/'manifest.json',manifest)
    dump(out/'split.json',load_split())
    dump(out/'few_shot_examples.json',examples)
    for name,items in [('input.jsonl',cases),('evaluation_references.jsonl',[refs[c['id']] for c in cases])]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf-8')
    tasks = [(c,n,a) for c in cases for n in range(repeats) for a in approaches]
    random.Random(seed).shuffle(tasks)
    rows = []
    for case,repeat,arm in tasks:
        if stop_requested and stop_requested():
            break
        row = {'case_id':case['id'],'repeat':repeat,'approach':arm,'status':'completed',
               'experiment_protocol':EXPERIMENT_PROTOCOL,'split':split,
               'evaluation_method':EVALUATION_METHOD,'reference_id':case['id'],
               'reference':refs[case['id']]['reference'],'metrics':{}}
        try:
            started = time.perf_counter()
            settings=METHODS[arm]
            result = generate_summary(case['query'],case['results'],approach='baseline',model=model,client=client,
                max_claims=max_claims,input_cost_per_million=input_price,output_cost_per_million=output_price,
                context_policy=BENCHMARK_CONTEXT_POLICY,include_review_metadata=True,as_of=clock,
                few_shot_examples=examples if settings['few_shot'] else None,
                compression_ratio=compression_ratio if settings['compression'] else None)
            result['generation_approach']='baseline'
            result['approach']=arm
            result['experiment_protocol']=EXPERIMENT_PROTOCOL
            full = build_context(case['results'],context_policy=BENCHMARK_CONTEXT_POLICY,
                                 include_review_metadata=True,as_of=clock)
            expected=compress_context(case['query'],full,compression_ratio)[0] if settings['compression'] else full
            actual=result['context_evidence']
            if actual != expected:
                raise ValueError('Model evidence differs from deterministic context construction')
            row['measured_execution_seconds'] = round(time.perf_counter()-started,6)
            row['result'] = result
            original_chars=sum(len(e['text']) for e in full)
            context_chars=sum(len(e['text']) for e in actual)
            metrics = score_summary(result['summary'],row['reference'],result['context_chunk_ids'])
            metrics.update(latency_seconds=result['latency_seconds'],model_calls=len(result['stage_calls']),
                prompt_tokens=result['usage'].get('prompt_tokens'),completion_tokens=result['usage'].get('completion_tokens'),
                estimated_cost_usd=result['estimated_cost_usd'],final_claims=len(result['claims']),
                provenance_rejected=result['validation']['rejected_claims'],
                context_preparation_seconds=result['context_preparation_seconds'],
                original_context_text_chars=original_chars,context_text_chars=context_chars,
                context_text_ratio=context_chars/original_chars if original_chars else 0,
                context_chunks=len(actual))
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
    parser.add_argument('--split',choices=('test','development'),default='test')
    parser.add_argument('--compression-ratio',type=float,default=DEFAULT_RATIO)
    parser.add_argument('--max-claims',type=int,default=4);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--approaches',nargs='+',choices=APPROACHES,default=APPROACHES)
    parser.add_argument('--timeout',type=int,default=180);parser.add_argument('--max-output-tokens',type=int,default=8192)
    parser.add_argument('--thinking',choices=('auto','disabled','enabled'),default='auto')
    parser.add_argument('--input-price',type=float);parser.add_argument('--output-price',type=float)
    parser.add_argument('--as-of',help='Fixed ISO UTC reference time for reproducible review-age metadata')
    args = parser.parse_args()
    cases=select_cases(jsonl(BENCHMARK),args.split)
    if not 1<=args.limit<=len(cases):
        parser.error(f'limit must be between 1 and {len(cases)} for {args.split}')
    client = OpenAICompatibleClient(base_url=args.base_url,timeout=args.timeout,max_tokens=args.max_output_tokens,thinking=args.thinking)
    if not client.api_key:
        parser.error('Set GENERATION_API_KEY or OPENAI_API_KEY in your own terminal')
    run(cases[:args.limit],args.output,client=client,model=args.model,approaches=args.approaches,
        repeats=args.repeats,max_claims=args.max_claims,seed=args.seed,input_price=args.input_price,
        output_price=args.output_price,as_of=args.as_of,split=args.split,compression_ratio=args.compression_ratio,
        config={k:getattr(args,k) for k in ('base_url','timeout','max_output_tokens','thinking')})


if __name__=='__main__':
    main()
