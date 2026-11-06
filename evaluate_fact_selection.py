"""Score structured fact selections by exact ID matching; no LLM or embedding model."""
import argparse
import collections
import json
from pathlib import Path
import random
import statistics

ANNOTATIONS = Path(__file__).resolve().parent / 'data/generation_annotations'
REFERENCE_VERSION = 'human_review_v1'


def paired_comparisons(rows, bootstrap_samples=5000, seed=42):
    """Paired method differences, bootstrapped over queries (repeats stay together)."""
    by={(r['case_id'],r['approach'],r.get('repeat',0)):r for r in rows if r['status']=='completed'}
    comparisons=[]
    for before,after in [('baseline','improved'),('improved','verify_delete'),('improved','verify_revise')]:
        for metric in ('fact_selection_precision','fact_selection_recall','fact_selection_f1',
                       'coverage_unit_recall','polarity_contrast_coverage'):
            differences=collections.defaultdict(list)
            for (qid,arm,repeat),a in by.items():
                b=by.get((qid,after,repeat))
                if arm!=before or b is None:continue
                x,y=a['metrics'].get(metric),b['metrics'].get(metric)
                if x is not None and y is not None:differences[qid].append(y-x)
            values=[statistics.mean(ds) for ds in differences.values()]
            if not values:continue
            rng=random.Random(seed)
            boots=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(bootstrap_samples))
            comparisons.append({'before':before,'after':after,'metric':metric,'n':len(values),
                'paired_rows':sum(map(len,differences.values())),
                'mean_delta':statistics.mean(values),
                'bootstrap_95_percent_CI':[boots[int(.025*bootstrap_samples)],boots[int(.975*bootstrap_samples)-1]],
                'wins':sum(v>1e-12 for v in values),'ties':sum(abs(v)<=1e-12 for v in values),
                'losses':sum(v< -1e-12 for v in values),'bootstrap_samples':bootstrap_samples,'seed':seed})
    return comparisons


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def score_selection(annotation, prediction):
    ids=prediction.get('selected_fact_ids')
    if not isinstance(ids,list) or any(not isinstance(x,str) or not x for x in ids):
        raise ValueError('selected_fact_ids must be a list of nonempty strings; null is an unfilled template')
    selected=set(ids);reference=set(annotation['reference_fact_ids']);context=set(annotation['context_fact_ids'])
    optional=set(annotation.get('optional_fact_ids',[]))
    excluded=set(annotation.get('explicit_excluded_fact_ids',[]))
    if reference & optional or reference & excluded or optional & excluded:
        raise ValueError('Required, optional and excluded fact sets must be disjoint')
    scored=selected-optional
    tp=len(scored & reference)
    if not reference and not scored:
        precision=recall=f1=None
    else:
        precision=tp/len(scored) if scored else 0.0
        recall=tp/len(reference) if reference else None
        f1=2*precision*recall/(precision+recall) if recall is not None and precision+recall else (0.0 if reference else None)
    units=annotation['coverage_units']
    covered=sum(bool(selected & set(u['supporting_fact_ids'])) for u in units)
    contrasts=[d for d in annotation['contrast_groups'] if d['required_for_query']]
    both=sum(bool(selected & set(d['positive_fact_ids'])) and bool(selected & set(d['negative_fact_ids'])) for d in contrasts)
    return {'fact_selection_precision':precision,'fact_selection_recall':recall,'fact_selection_f1':f1,
        'coverage_unit_recall':covered/len(units) if units else None,
        'polarity_contrast_coverage':both/len(contrasts) if contrasts else None,
        'out_of_context_fact_rate':len(selected-context)/len(selected) if selected else None,
        'selected_fact_count':len(scored),'reference_fact_count':len(reference),
        'raw_selected_fact_count':len(selected),'optional_selected_fact_count':len(selected & optional),
        'explicit_excluded_selected_fact_count':len(selected & excluded),
        'scored_false_positive_count':len(scored)-tp,
        'correct_selected_fact_count':tp,'coverage_unit_count':len(units),'covered_unit_count':covered,
        'required_contrast_count':len(contrasts),'covered_contrast_count':both,
        'duplicate_fact_id_count':len(ids)-len(selected),
        'empty_reference_correct_abstention':int(not scored) if not reference else None,
        'empty_core_reference_correct_abstention':int(not scored) if not reference else None,
        'condition_retention_rate':None}


def evaluate(annotations, predictions):
    targets={a['id']:a for a in annotations}
    if len(targets)!=len(annotations):raise ValueError('Duplicate annotation case IDs')
    keys=set();rows=[]
    for p in predictions:
        key=(p['id'],p.get('approach','provided'),p.get('repeat',0))
        if key in keys:raise ValueError('Duplicate prediction case/approach/repeat')
        keys.add(key)
        if p['id'] not in targets:raise ValueError('Unknown prediction case ID')
        rows.append({'id':p['id'],'approach':key[1],'repeat':key[2],
                     'metrics':score_selection(targets[p['id']],p)})
    if not rows:raise ValueError('No predictions')
    groups=collections.defaultdict(list)
    for r in rows:groups[r['approach']].append(r)
    aggregates={}
    for arm,group in groups.items():
        result={'prediction_rows':len(group),'unique_queries':len({r['id'] for r in group}),
                'dataset_queries':len(targets),'query_coverage':len({r['id'] for r in group})/len(targets)}
        for metric in group[0]['metrics']:
            values=[r['metrics'][metric] for r in group if r['metrics'][metric] is not None]
            result[metric]={'mean':statistics.mean(values) if values else None,'n':len(values)}
        tp=sum(r['metrics']['correct_selected_fact_count'] for r in group)
        chosen=sum(r['metrics']['selected_fact_count'] for r in group)
        gold=sum(r['metrics']['reference_fact_count'] for r in group)
        result['micro_fact_precision']=tp/chosen if chosen else None
        result['micro_fact_recall']=tp/gold if gold else None
        aggregates[arm]=result
    return {'rows':rows,'aggregates':aggregates,'uses_model':False,
        'reference_version':REFERENCE_VERSION,
        'scope':'Exact-ID selection against aspect-label targets with query-specific human review.',
        'condition_note':'Condition retention has not been annotated and is not scored.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',help='JSONL: id, selected_fact_ids; optional approach, repeat, summary')
    parser.add_argument('--annotations',default=str(ANNOTATIONS/'query_annotations.jsonl'))
    parser.add_argument('--output',default='fact_selection_results.json')
    args=parser.parse_args()
    report=evaluate(read_jsonl(args.annotations),read_jsonl(args.predictions))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report['aggregates'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
