"""Structured fact selection experiment: generation calls only; exact-ID scoring."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from evaluate_fact_selection import ANNOTATIONS, REFERENCE_VERSION, paired_comparisons, score_selection
from rag_generator import APPROACHES, PROMPTS, OpenAICompatibleClient, build_context, generate_summary, safe_error

ROOT=Path(__file__).resolve().parent
BENCHMARK=ROOT/'experiments/steam_generation_eval.jsonl'
ASPECTS=('story','graphics_visuals','gameplay','controls','performance','audio','bugs_stability','online_multiplayer')
# These references and archived report scores were built from full review texts.
BENCHMARK_CONTEXT_POLICY='full_review'


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def benchmark(cases):
    references={r['id']:r for r in jsonl(ANNOTATIONS/'query_annotations.jsonl')}
    original={c['id']:c for c in jsonl(BENCHMARK)}
    # Only identifiers are used to build model input; no aspect labels or gold spans.
    samples={p['chunk_id']:p['sample_id'] for p in jsonl(ANNOTATIONS/'passages.jsonl')}
    for c in cases:
        expected=original.get(c['id'])
        signature=lambda x:(x['query'],[(h['chunk_id'],h['text'],(h.get('metadata') or {}).get('game_name')) for h in x['results']])
        if expected is None or signature(c)!=signature(expected):
            raise ValueError('Uploaded case does not match the fixed annotated benchmark')
        context=build_context(c['results'],context_policy=BENCHMARK_CONTEXT_POLICY)
        if [x['chunk_id'] for x in context]!=references[c['id']]['context_chunk_ids']:
            raise ValueError('Constructed context does not match the reference context')
    return references,samples


def catalog(context,samples):
    # Do NOT filter using human Not mentioned labels: that would leak aspect presence.
    return [{'fact_id':'F_'+samples[h['chunk_id']]+'_'+aspect,'chunk_id':h['chunk_id'],'aspect':aspect}
            for h in context for aspect in ASPECTS]


class SharedDraft:
    def __init__(self,client,cache,key):self.client,self.cache,self.key=client,cache,key;self.replayed=None
    def complete(self,messages,model,temperature):
        draft=messages[0]['content'].startswith((PROMPTS/'improved.txt').read_text())
        if draft and self.key in self.cache:
            self.replayed=self.cache[self.key]
            return self.replayed['response']
        start=time.perf_counter();response=self.client.complete(messages,model,temperature)
        if draft:self.cache[self.key]={'response':response,'latency_seconds':time.perf_counter()-start}
        return response


def aggregate(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['approach']].append(r)
    table=[]
    for arm,group in groups.items():
        valid=[r for r in group if r['status']=='completed']
        out={'approach':arm,'attempted':len(group),'completed':len(valid),'failed':len(group)-len(valid)}
        for k in sorted({k for r in valid for k in r['metrics']}):
            values=[r['metrics'][k] for r in valid if r['metrics'].get(k) is not None]
            out[k]=statistics.mean(values) if values else None;out[k+'_n']=len(values)
        latency=sorted(r['metrics']['latency_seconds'] for r in valid)
        for p in (.5,.95):out['p'+str(int(p*100))+'_latency_seconds']=latency[round((len(latency)-1)*p)] if latency else None
        for k in ('prompt_tokens','completion_tokens','estimated_cost_usd'):
            values=[r['metrics'].get(k) for r in valid]
            out['total_'+k]=sum(values) if values and all(v is not None for v in values) else None
        table.append(out)
    return table


def export(out,rows):
    table=aggregate(rows)
    paired=paired_comparisons(rows)
    dump(out/'generation_results.json',rows)
    dump(out/'report_data.json',{'evaluation_method':'exact_fact_id_matching_human_review_v1',
        'reference_version':REFERENCE_VERSION,'aggregates':table,'paired_comparisons':paired,
        'scope':'Required/optional/excluded review-aspect selection. Condition retention is not scored.'})
    fields=sorted({k for r in table for k in r})
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(table)
    predictions=[{'id':r['case_id'],'approach':r['approach'],'repeat':r['repeat'],
        'selected_fact_ids':r['result']['selected_fact_ids'],'summary':r['result']['summary']} for r in rows if r['status']=='completed']
    (out/'predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in predictions),encoding='utf-8')
    lines=['# Structured fact selection results','','Scoring uses exact ID matching; no evaluator model is called.',
        'All eight aspect slots per review are visible to generation; gold labels and reference sets are not.',
        'P/R/F1 concern review-aspect selection. Natural-language summary entailment is not scored.',
        'References include 40 query-specific human decisions. Optional background is not required or penalized.',
        'Condition retention has not been annotated and is not scored.',
        'C/D self-verification uses the generator model and is part of generation, not the final scorer.','',
        '| Method | Completed | Failed | Precision | Recall | F1 | Unit recall | Contrast coverage | Latency s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append('| '+' | '.join(str(r.get(k,'N/A')) for k in ('approach','completed','failed','fact_selection_precision','fact_selection_recall','fact_selection_f1','coverage_unit_recall','polarity_contrast_coverage','latency_seconds'))+' |')
    lines.extend(['','All metric means disclose valid counts in summary.csv. Failed tasks are listed separately.',
        'B/C/D share a draft; logical latency, usage and cost include one draft per arm.'])
    lines.extend(['','## Paired F1 comparisons','','| After - before | Queries | Delta (pp) | Bootstrap 95% CI (pp) |',
                  '|---|---:|---:|---:|'])
    for p in paired:
        if p['metric']=='fact_selection_f1':
            lo,hi=p['bootstrap_95_percent_CI']
            lines.append(f"| {p['after']} - {p['before']} | {p['n']} | {100*p['mean_delta']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] |")
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return table


def run(cases,out,*,client,model,approaches=APPROACHES,repeats=1,max_claims=4,seed=42,
        input_price=None,output_price=None,config=None,progress=None,stop_requested=None):
    references,samples=benchmark(cases)
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(config or {},evaluation_method='exact_fact_id_matching_human_review_v1',
        reference_version=REFERENCE_VERSION,model=model,context_policy=BENCHMARK_CONTEXT_POLICY,
        approaches=list(approaches),repeats=repeats,max_claims=max_claims,seed=seed,
        candidate_catalog='all_eight_aspects_for_each_visible_review',reference_labels_in_prompt=False,
        context_limits={'max_context_chars':24000,'max_chunk_chars':4000},
        prompt_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROMPTS.glob('*.txt')},
        data_hash=hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        code_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('rag_generator.py','review_context.py','run_fact_experiments.py','evaluate_fact_selection.py')},
        passage_mapping_hash=hashlib.sha256((ANNOTATIONS/'passages.jsonl').read_bytes()).hexdigest(),
        annotation_hash=hashlib.sha256((ANNOTATIONS/'query_annotations.jsonl').read_bytes()).hexdigest(),
        human_review_hash=hashlib.sha256((ANNOTATIONS/'human_review.json').read_bytes()).hexdigest())
    dump(out/'manifest.json',manifest)
    write=''.join(json.dumps(references[c['id']],ensure_ascii=False)+'\n' for c in cases)
    (out/'evaluation_references.jsonl').write_text(write,encoding='utf-8')
    tasks=[(c,n,a) for c in cases for n in range(repeats) for a in approaches]
    random.Random(seed).shuffle(tasks);cache={};rows=[]
    for case,repeat,arm in tasks:
        if stop_requested and stop_requested():break
        row={'case_id':case['id'],'repeat':repeat,'approach':arm,'status':'completed','metrics':{}}
        try:
            context=build_context(case['results'],context_policy=BENCHMARK_CONTEXT_POLICY);candidates=catalog(context,samples)
            transport=SharedDraft(client,cache,(case['id'],repeat))
            result=generate_summary(case['query'],case['results'],approach=arm,model=model,client=transport,
                max_claims=max_claims,fact_candidates=candidates,input_cost_per_million=input_price,
                output_cost_per_million=output_price,context_policy=BENCHMARK_CONTEXT_POLICY)
            row['measured_execution_seconds']=result['latency_seconds']
            if transport.replayed:
                first=result['stage_calls'][0]
                result['latency_seconds']+=transport.replayed['latency_seconds']-first['latency_seconds']
                first['latency_seconds']=transport.replayed['latency_seconds'];first['shared_draft']=True
            row['result']=result
            metrics=score_selection(references[case['id']],{'selected_fact_ids':result['selected_fact_ids']})
            candidate_ids={c['fact_id'] for c in candidates}
            metrics.update(latency_seconds=result['latency_seconds'],model_calls=len(result['stage_calls']),
                prompt_tokens=result['usage'].get('prompt_tokens'),completion_tokens=result['usage'].get('completion_tokens'),
                estimated_cost_usd=result['estimated_cost_usd'],final_claims=len(result['claims']),
                unknown_candidate_id_count=len(set(result['selected_fact_ids'])-candidate_ids),
                provenance_rejected=result['validation']['rejected_claims'])
            row['metrics']=metrics
        except Exception as error:
            row['status']='failed';row.update(safe_error(error))
        with (out/'checkpoint.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        rows.append(row)
        if progress:progress(row,len(rows),len(tasks))
    export(out,rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--model',default='deepseek-v4-pro');p.add_argument('--base-url',default='https://api.deepseek.com')
    p.add_argument('--limit',type=int,default=3);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--max-claims',type=int,default=4);args=p.parse_args()
    if not 1<=args.limit<=100 or not 1<=args.repeats<=5 or not 1<=args.max_claims<=12:p.error('Invalid size parameters')
    client=OpenAICompatibleClient(base_url=args.base_url)
    run(jsonl(BENCHMARK)[:args.limit],args.output,client=client,model=args.model,repeats=args.repeats,max_claims=args.max_claims,config={'base_url':args.base_url})

if __name__=='__main__':main()
