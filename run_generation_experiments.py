"""Reproducible, resumable generation experiments. See EXPERIMENTS.md."""
import argparse
import collections
import csv
import hashlib
import itertools
import math
import json
import os
from pathlib import Path
import random
import statistics
import time

from rag_generator import APPROACHES, PROMPTS, GenerationError, OpenAICompatibleClient, build_context, generate_summary, safe_error
from evaluate_generation import judge_result

PIN = '34e8675a6af11e6871119d2de264a2487ba9b7b5'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def prepare(args):
    root = Path(args.retrieval_repo)
    runs = json.loads((root / 'eval/results/runs.json').read_text())
    qrels = json.loads((root / 'eval/results/qrels.json').read_text())
    records = json.loads((root / 'data/steam_annotations_confirmed.json').read_text())['records']
    # Rankings in the pinned repository are condensed to annotated chunks, NOT production rankings.
    chunks = {r['chunk_id']: {'chunk_id': r['chunk_id'], 'text': r['text'],
               'metadata': {'game_name': r['game_name']}} for r in records}
    for r in jsonl(root / 'chunks.jsonl'):
        if r['id'] in chunks:
            if r['text'] != chunks[r['id']]['text']:
                raise ValueError('Annotation/corpus text mismatch: ' + r['id'])
            chunks[r['id']]['metadata'].update(r.get('metadata', {}))
    cases = []
    for method in args.retrievers:
        for q in jsonl(root / 'eval/queries.jsonl'):
            qid = q['query_id']
            ids = runs[method][qid][:args.top_k]
            cases.append({'id': method + '/' + qid, 'query_id': qid, 'query': q['query_text'],
                'query_type': q['query_type'], 'retriever': method, 'results': [chunks[c] for c in ids],
                'relevance_grades': {c: qrels[qid].get(c, 0) for c in ids},
                'source': {'commit': PIN, 'ranking': 'condensed_annotated_benchmark',
                           'relevance': 'derived_from_aspect_annotations_not_direct_query_judgments'},
                'reference': None, 'key_points': []})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('Refusing to overwrite an existing dataset')
    out.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
    save(out.with_suffix('.audit.json'), {'cases': len(cases), 'annotated_records': len(records),
        'unique_texts': len({r['text'] for r in records}),
        'source_file_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('eval/queries.jsonl','eval/results/runs.json','eval/results/qrels.json','data/steam_annotations_confirmed.json','chunks.jsonl')},
        'independent_annotator_agreement': None,
        'note': 'Confirmed status does not establish inter-annotator agreement. Add human references/key_points; never fabricate them.'})
    print('Prepared', len(cases), 'cases:', out)


class DraftClient:
    """Reuse the identical structured draft across B/C/D, including after resume."""
    def __init__(self, transport, path):
        self.transport, self.path, self.replayed = transport, path, None

    def complete(self, messages, model, temperature):
        is_draft = messages[0]['content'] == (PROMPTS / 'improved.txt').read_text(encoding='utf-8')
        if is_draft and self.path.exists():
            cached = json.loads(self.path.read_text())
            self.replayed = cached
            return cached['response']
        start = time.perf_counter()
        response = self.transport.complete(messages, model, temperature)
        if is_draft:
            save(self.path, {'response': response, 'latency_seconds': time.perf_counter() - start})
        return response


def context_metrics(case, context):
    grades = case.get('relevance_grades')
    if grades is None:
        return {}
    known = [grades[x['chunk_id']] for x in context if x['chunk_id'] in grades]
    if any(type(v) is not int or v < 0 for v in known):
        raise ValueError('Relevance grades must be nonnegative integers')
    return {'context_judgment_coverage': len(known) / len(context) if context else None,
            'context_relevance_fraction': sum(g > 0 for g in known) / len(known) if known else None,
            'context_ranked_precision_ap': sum(sum(v > 0 for v in known[:i+1]) / (i+1)
                 for i,g in enumerate(known) if g > 0) / sum(g > 0 for g in known) if any(g > 0 for g in known) else (0 if known else None)}


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    return values[round((len(values)-1)*p)]


def summarize(rows, seed=42):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['retriever'], row['approach'])].append(row)
    table = []
    for (retriever, arm), group in sorted(groups.items()):
        ok = [r for r in group if r['status'] == 'completed']
        metrics = sorted({k for r in ok for k in r['metrics']})
        item = {'retriever': retriever, 'approach': arm, 'attempted': len(group), 'completed': len(ok),
                'failed': len(group)-len(ok)}
        for metric in metrics:
            values = [r['metrics'][metric] for r in ok if r['metrics'].get(metric) is not None]
            item[metric] = statistics.mean(values) if values else None
            item[metric + '_n'] = len(values)
        for metric in ('estimated_cost_usd','estimated_judge_cost_usd','prompt_tokens','completion_tokens','judge_prompt_tokens','judge_completion_tokens'):
            values=[r['metrics'].get(metric) for r in ok]
            item['total_'+metric]=sum(values) if values and all(v is not None for v in values) else None
        for metric in ('latency_seconds',):
            values = [r['metrics'][metric] for r in ok]
            item['p50_' + metric], item['p95_' + metric] = percentile(values,.5), percentile(values,.95)
            item['serial_records_per_second'] = len(values)/sum(values) if sum(values) else None
        table.append(item)
    paired = []
    rng = random.Random(seed)
    for retriever in sorted({r['retriever'] for r in rows}):
        for a,b in itertools.combinations(APPROACHES,2):
            left = {(r['case_id'],r['repeat']):r for r in rows if r['retriever']==retriever and r['approach']==a and r['status']=='completed'}
            right = {(r['case_id'],r['repeat']):r for r in rows if r['retriever']==retriever and r['approach']==b and r['status']=='completed'}
            keys = sorted(left.keys() & right.keys())
            for metric in ('rougeL_f1','bertscore_f1','faithfulness','relevance','key_point_coverage','latency_seconds','estimated_cost_usd'):
                # Cluster repeated runs within each query before bootstrapping query units.
                units = collections.defaultdict(list)
                for key in keys:
                    x,y=left[key]['metrics'].get(metric),right[key]['metrics'].get(metric)
                    if x is not None and y is not None:
                        units[key[0]].append(y-x)
                values=[statistics.mean(v) for v in units.values()]
                if values:
                    boots=[statistics.mean(rng.choices(values,k=len(values))) for _ in range(1000)]
                    paired.append({'retriever':retriever,'comparison':b+' minus '+a,'metric':metric,
                        'queries':len(values),'mean_delta':statistics.mean(values),
                        'ci95':[percentile(boots,.025),percentile(boots,.975)]})
    by_type={}
    for kind in sorted({r.get('query_type','unspecified') for r in rows}):
        subset=[r for r in rows if r.get('query_type','unspecified')==kind]
        by_type[kind]={'attempted':len(subset),'completed':sum(r['status']=='completed' for r in subset)}
    return {'aggregates':table,'paired_query_bootstrap':paired,'query_type_counts':by_type}


def export_report(out, rows, seed):
    report=summarize(rows,seed)
    save(out/'report_data.json',report)
    table=report['aggregates']
    fields=sorted({k for r in table for k in r})
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    save(out/'generation_results.json',rows)
    text=['# Generation experiment report','',
        'Scores are estimates unless explicitly human-labeled. Missing metrics are unavailable, never zero.',
        'Cached B/C/D drafts are shared. Latency/token/cost metrics include the logical draft cost for every arm.',
        'Cached benchmark rankings are condensed to annotated chunks; these are not production retrieval latency/results.',
        'Bootstrap intervals use queries as units; one-query intervals cannot establish significance.','',
        '| Retriever | Arm | Completed | Failed | Faithfulness (1–5) | Relevance (1–5) | ROUGE-L | Latency s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        text.append('| '+' | '.join(str(r.get(k,'N/A')) for k in ['retriever','approach','completed','failed','faithfulness','relevance','rougeL_f1','latency_seconds'])+' |')
    text += ['', 'See summary.csv for coverage counts, context metrics, refusal, token use, repair, P50/P95 and costs.',
             'See report_data.json for paired differences and confidence intervals. Human agreement requires separate independent annotations.']
    with (out/'human_claim_audit.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['record_id','case_id','approach','repeat','claim','chunk_ids','evidence','annotator','label'])
        for row in rows:
            for i,c in enumerate(row.get('result',{}).get('claims',[])):
                writer.writerow([digest([row['case_id'],row['approach'],row['repeat'],i]),row['case_id'],row['approach'],row['repeat'],c['text'],json.dumps(c['chunk_ids']),json.dumps(c['evidence'],ensure_ascii=False),'',''])
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


def run(args, client=None, judge_client=None, *, progress=None, stop_requested=None):
    cases=jsonl(args.dataset)
    if args.limit: cases=cases[:args.limit]
    if not cases or any(not isinstance(c.get('id'),str) or not c['id'].strip() for c in cases) or len({c['id'] for c in cases})!=len(cases):
        raise ValueError('Dataset must contain unique nonempty case IDs')
    for c in cases:
        build_context(c['results'],args.max_context_chars,args.max_chunk_chars)
        if not isinstance(c.get('query'),str) or not c['query'].strip(): raise ValueError('Missing query')
        if c.get('reference') is not None and (not isinstance(c['reference'],str) or not c['reference'].strip()): raise ValueError('Invalid reference')
        if not isinstance(c.get('key_points',[]),list) or any(not isinstance(x,str) or not x.strip() for x in c.get('key_points',[])): raise ValueError('Invalid key points')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if k not in ('api_key_file','judge_key_file','output','command','limit')}
    config['effective_base_url']=args.base_url or os.getenv('GENERATION_BASE_URL','https://api.openai.com/v1')
    config['effective_judge_base_url']=args.judge_base_url or config['effective_base_url']
    config['dataset_hash']=digest(cases)
    config['prompt_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PROMPTS.glob('*.txt')}
    config['code_hashes']={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in ('rag_generator.py','evaluate_generation.py','run_generation_experiments.py')}
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise ValueError('Configuration/data/code changed; use a new output directory')
    save(manifest,config)
    tasks=[(c,n,a) for c in cases for n in range(args.repeats) for a in args.approaches]
    random.Random(args.seed).shuffle(tasks)
    print('Cases:',len(cases),'tasks:',len(tasks),'upper bound generation calls:',sum(1 if a in ('baseline','improved') else 1+args.max_claims*(3 if a=='verify_revise' else 1) for _,_,a in tasks))
    save(out/'readiness.json',{'cases':len(cases),
        'human_references':sum(bool(c.get('reference')) for c in cases),
        'human_key_points':sum(bool(c.get('key_points')) for c in cases),
        'context_labels':sum('relevance_grades' in c for c in cases),
        'independent_judge_model':bool(args.judge_model and args.judge_model!=args.model),
        'annotation_agreement':'Not established by this experiment; supply independent annotation logs.',
        'retrieval_latency':'Not measured from saved rankings; benchmark the live retriever separately.'})
    if args.preflight:
        save(out/'preflight.json',{'tasks':len(tasks),'references':sum(bool(c.get('reference')) for c in cases),
            'key_points':sum(bool(c.get('key_points')) for c in cases),'context_labeled':sum('relevance_grades' in c for c in cases),
            'judge_enabled':bool(args.judge_model),'note':'No API calls. Missing human references/key points cannot be computed from aspect labels.'})
        return []
    if args.api_key_file: os.environ['GENERATION_API_KEY']=Path(args.api_key_file).read_text().strip()
    transport=client or OpenAICompatibleClient(base_url=args.base_url,timeout=args.timeout,max_tokens=args.max_output_tokens,thinking=args.thinking)
    judge_transport=judge_client or OpenAICompatibleClient(base_url=args.judge_base_url or args.base_url,
        api_key=Path(args.judge_key_file).read_text().strip() if args.judge_key_file else os.getenv('JUDGE_API_KEY') or None,timeout=args.timeout,max_tokens=args.max_output_tokens,thinking=args.thinking)
    rouge=None
    if 'rouge' in args.reference_metrics:
        from rouge_score.rouge_scorer import RougeScorer
        rouge=RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=True)
    scorer=None
    if 'bertscore' in args.reference_metrics:
        from bert_score import BERTScorer
        scorer=BERTScorer(model_type=args.bertscore_model,lang='en',device=args.device)
    checkpoint=out/'checkpoint.jsonl'
    rows=jsonl(checkpoint) if checkpoint.exists() else []
    done={(r['case_id'],r['repeat'],r['approach']) for r in rows}
    for case,repeat,arm in tasks:
        key=(case['id'],repeat,arm)
        if key in done: continue
        if stop_requested and stop_requested(): break
        row={'case_id':case['id'],'repeat':repeat,'approach':arm,'retriever':case.get('retriever','provided'),
             'query_type':case.get('query_type','unspecified'),'status':'completed','metrics':{}}
        try:
            dc=DraftClient(transport,out/('draft_'+digest([case['id'],repeat])+'.json'))
            result=generate_summary(case['query'],case['results'],approach=arm,model=args.model,client=dc,
                max_claims=args.max_claims,max_context_chars=args.max_context_chars,max_chunk_chars=args.max_chunk_chars,
                input_cost_per_million=args.input_price,output_cost_per_million=args.output_price)
            row['measured_execution_seconds']=result['latency_seconds']
            if dc.replayed:
                first=result['stage_calls'][0]
                result['latency_seconds']+=dc.replayed['latency_seconds']-first['latency_seconds']
                first['latency_seconds']=dc.replayed['latency_seconds'];first['shared_draft']=True
            row['result']=result
            m=row['metrics']; v=result['validation']
            m.update(context_metrics(case,build_context(case['results'],args.max_context_chars,args.max_chunk_chars)))
            m.update(latency_seconds=result['latency_seconds'],model_calls=len(result['stage_calls']),
                estimated_cost_usd=result['estimated_cost_usd'],abstention=int(result['status']!='ok'),
                final_claims=len(result['claims']),revised_claims=v['revised_claims'],discarded_claims=v['semantic_discarded_claims'],
                provenance_rejected=v['rejected_claims'],prompt_tokens=result['usage'].get('prompt_tokens'),
                completion_tokens=result['usage'].get('completion_tokens'))
            pred=' '.join(c['text'] for c in result['claims']) or result['summary']
            if case.get('reference'):
                if rouge: m.update({k+'_f1':v.fmeasure for k,v in rouge.score(case['reference'],pred).items()})
                if scorer: m['bertscore_f1']=scorer.score([pred],[case['reference']])[2][0].item()
            if args.judge_model:
                # Judge sees exactly the evidence available to generation, including custom truncation.
                exact=[{'chunk_id':x['chunk_id'],'text':x['text'],'metadata':{'game_name':x['game_name'],'source_url':x['source_url']}}
                       for x in build_context(case['results'],args.max_context_chars,args.max_chunk_chars)]
                judge_started=time.perf_counter()
                judge=judge_result(case['query'],result,exact,judge_transport,args.judge_model,case.get('key_points'),
                    context=build_context(case['results'],args.max_context_chars,args.max_chunk_chars),detailed=True)
                row['judge']=judge
                row['judge_latency_seconds']=time.perf_counter()-judge_started
                ju=judge['usage']
                m['judge_prompt_tokens']=ju.get('prompt_tokens');m['judge_completion_tokens']=ju.get('completion_tokens')
                m['estimated_judge_cost_usd']=None
                if args.judge_input_price is not None and args.judge_output_price is not None and all(type(ju.get(k)) is int and ju[k]>=0 for k in ('prompt_tokens','completion_tokens')):
                    m['estimated_judge_cost_usd']=(ju['prompt_tokens']*args.judge_input_price+ju['completion_tokens']*args.judge_output_price)/1e6
                for label,name in [('claim_support','claim_faithfulness_fraction'),('citation_support','citation_precision'),('context_relevance','judge_context_relevance_fraction')]:
                    vals=judge['scores'][label]
                    m[name]=statistics.mean(vals) if vals else None
                for k in ('faithfulness','relevance'): m[k]=judge['scores'][k]['score']
                points=judge['scores'].get('coverage',[])
                if points: m['key_point_coverage']=statistics.mean(p['covered'] for p in points)
        except Exception as error:
            # Never serialize raw API exceptions, credentials, or request headers.
            row['status']='failed';row.update(safe_error(error))
        with checkpoint.open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
        rows.append(row)
        if progress: progress(row, len(rows), len(tasks))
        print(len(rows),'/',len(tasks),case['id'],arm,row['status'],flush=True)
    export_report(out,rows,args.seed)
    return rows


def agreement(args):
    """Independent long-form CSV: record_id,annotator,label (one label per item)."""
    entries=collections.defaultdict(dict)
    with open(args.annotations,newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not all(row.get(k) for k in ('record_id','annotator','label')): raise ValueError('Missing annotation field')
            if row['annotator'] in entries[row['record_id']]: raise ValueError('Duplicate annotator/item')
            entries[row['record_id']][row['annotator']]=row['label']
    pairs=collections.defaultdict(list)
    for labels in entries.values():
        for a,b in itertools.combinations(sorted(labels),2):pairs[(a,b)].append((labels[a],labels[b]))
    results=[]
    for pair,values in pairs.items():
        n=len(values);observed=sum(a==b for a,b in values)/n
        ca,cb=collections.Counter(a for a,b in values),collections.Counter(b for a,b in values)
        expected=sum(ca[k]*cb[k] for k in ca.keys()|cb.keys())/n**2
        results.append({'annotators':pair,'overlapping_records':n,'agreement':observed,
                        'cohen_kappa':(observed-expected)/(1-expected) if expected<1 else None})
    save(args.output,{'unique_records':len(entries),'double_labeled_records':sum(len(v)>=2 for v in entries.values()),'pairs':results})


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--retrieval-repo',required=True);a.add_argument('--retrievers',nargs='+',default=['hybrid_rerank']);a.add_argument('--top-k',type=int,default=10);a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('dataset');a.add_argument('--output',required=True)
    for name in ('api-key-file','judge-key-file','base-url','judge-base-url','judge-model'):a.add_argument('--'+name)
    a.add_argument('--model',default=os.getenv('GENERATION_MODEL','gpt-4o-mini'))
    a.add_argument('--approaches',nargs='+',choices=APPROACHES,default=list(APPROACHES))
    a.add_argument('--reference-metrics',nargs='*',choices=['rouge','bertscore'],default=[])
    a.add_argument('--bertscore-model',default='roberta-large');a.add_argument('--device',default='cpu')
    for name,default in [('repeats',1),('seed',42),('max-claims',6),('max-context-chars',24000),('max-chunk-chars',4000),('timeout',180),('max-output-tokens',8192)]:a.add_argument('--'+name,type=int,default=default)
    a.add_argument('--thinking',choices=['auto','disabled','enabled'],default='auto')
    a.add_argument('--limit',type=int);a.add_argument('--preflight',action='store_true')
    a.add_argument('--input-price',type=float);a.add_argument('--output-price',type=float)
    a.add_argument('--judge-input-price',type=float);a.add_argument('--judge-output-price',type=float)
    a=sub.add_parser('agreement');a.add_argument('annotations');a.add_argument('--output',required=True)
    return p


def main():
    args=parser().parse_args()
    for field in ('repeats','top_k','limit','timeout'):
        value=getattr(args,field,None)
        if value is not None and value<1: raise SystemExit(field+' must be positive')
    for field in ('input_price','output_price','judge_input_price','judge_output_price'):
        value=getattr(args,field,None)
        if value is not None and (not math.isfinite(value) or value<0):raise SystemExit(field+' must be finite and nonnegative')
    if args.command=='run' and len(set(args.approaches))!=len(args.approaches):raise SystemExit('Duplicate approaches')
    {'prepare':prepare,'run':run,'agreement':agreement}[args.command](args)

if __name__=='__main__': main()
