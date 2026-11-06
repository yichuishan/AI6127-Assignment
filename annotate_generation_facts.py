"""Derive auditable fact IDs from the pinned human aspect labels; no LLM calls."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

PIN = '34e8675a6af11e6871119d2de264a2487ba9b7b5'
ASPECTS = ('story','graphics_visuals','gameplay','controls','performance','audio','bugs_stability','online_multiplayer')
SENTIMENTS = {'Positive','Negative','Mixed','Neutral','Unclear','Not mentioned'}
NAMES = {'story':'story and characters','graphics_visuals':'graphics and visuals','gameplay':'gameplay',
         'controls':'controls','performance':'performance and optimization','audio':'music and audio',
         'bugs_stability':'bugs and stability','online_multiplayer':'online and multiplayer'}
PATTERNS = {
    'hardware':r'\b(?:(?:RTX|GTX)\s*\d{3,4}(?:\s*(?:Ti|Super))?|Ryzen\s*[3579](?:\s+\d{4}[A-Z]*)?|i[3579][- ]\d{4,5}[A-Z]*|\d+\s*GB\s*(?:RAM|VRAM))\b',
    'platform':r'\b(?:Steam\s*Deck|Linux|Windows\s*(?:10|11)?|PlayStation\s*[345]?|PS[345]|Xbox(?:\s+(?:One|Series\s*[XS]))?|on\s+(?:my\s+)?PC)\b',
    'play_mode':r'\b(?:single[- ]player|multi[- ]player|co[- ]?op(?:erative)?|solo|offline|online)\b',
    'time_or_version':r'\b(?:(?:after|before|since)\s+(?:(?:the|a|an|latest|recent|new|last|major|first)\s+){0,3}(?:patch|update|launch|release)|at\s+launch|on\s+release|(?:version|patch)\s*\d+(?:\.\d+)*|(?:after|for)\s+\d+(?:\.\d+)?\s+hours?)\b',
    'frequency':r'\b(?:occasionally|sometimes|rarely|frequently|always|every\s+time)\b',
}


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def stable_id(prefix, values):
    return prefix+'_'+hashlib.sha256(json.dumps(values,ensure_ascii=False).encode()).hexdigest()[:16]


def match_quote(text, quote):
    """Return exact source offsets, allowing whitespace-only normalization with a map."""
    if not quote or not quote.strip():
        return {'status':'missing','quote':None,'start':None,'end':None}
    at=text.find(quote)
    if at>=0:
        return {'status':'exact','quote':text[at:at+len(quote)],'start':at,'end':at+len(quote)}
    chars, mapping=[],[]
    for m in re.finditer(r'\s+|\S',text):
        chars.append(' ' if m.group().isspace() else m.group());mapping.append((m.start(),m.end()))
    normalized=''.join(chars);needle=' '.join(quote.split());at=normalized.find(needle)
    if at>=0:
        start,end=mapping[at][0],mapping[at+len(needle)-1][1]
        return {'status':'whitespace_aligned','quote':text[start:end],'start':start,'end':end}
    return {'status':'unmatched','quote':None,'start':None,'end':None}


def condition_candidates(record):
    text=record['text'];out=[]
    for kind,pattern in PATTERNS.items():
        for m in re.finditer(pattern,text,re.I):
            start=max(text.rfind('\n',0,m.start()),text.rfind('.',0,m.start()))+1
            ends=[p for p in (text.find('\n',m.end()),text.find('.',m.end())) if p>=0]
            end=min(ends)+1 if ends else len(text)
            out.append({'condition_id':stable_id('C',[record['chunk_id'],kind,m.start(),m.end()]),
                'chunk_id':record['chunk_id'],'kind':kind,'mention':m.group(),
                'start':m.start(),'end':m.end(),'surrounding_text':text[start:end].strip(),
                'annotation_source':'regex_candidate','human_reviewed':False,
                'importance':None,'applies_to_fact_ids':[]})
    return sorted(out,key=lambda x:(x['start'],x['kind']))


def fact_is_target(fact, query):
    return ((not query['target_games'] or fact['game_name'] in query['target_games'])
        and fact['aspect'] in query['target_aspects']
        and (query['target_sentiment']=='any' or fact['sentiment']==query['target_sentiment']))


def build(annotations, queries, generation_cases, output):
    records=annotations['records']
    if len({r['chunk_id'] for r in records})!=len(records) or len({r['sample_id'] for r in records})!=len(records):
        raise ValueError('Duplicate source chunk/sample IDs')
    out=Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Refusing to overwrite annotation output')
    out.mkdir(parents=True,exist_ok=True)
    facts,passages,conditions=[],[],[]
    for r in sorted(records,key=lambda x:x['sample_id']):
        if r['status']!='Confirmed':raise ValueError('Source contains unconfirmed labels')
        candidates=condition_candidates(r);local=[]
        for aspect in ASPECTS:
            label=r['human_annotation'][aspect];sentiment=label['sentiment']
            if sentiment not in SENTIMENTS:raise ValueError('Unexpected sentiment label')
            if sentiment=='Not mentioned':continue
            span=match_quote(r['text'],label.get('evidence',''))
            fid='F_'+r['sample_id']+'_'+aspect
            linked=[]
            if span['start'] is not None:
                linked=[c['condition_id'] for c in candidates if span['start']<=c['start'] and c['end']<=span['end']]
                for c in candidates:
                    if c['condition_id'] in linked:c['applies_to_fact_ids'].append(fid)
            fact={'fact_id':fid,'chunk_id':r['chunk_id'],'document_id':r['document_id'],
                'sample_id':r['sample_id'],'game_name':r['game_name'],'aspect':aspect,
                'sentiment':sentiment,'subjectivity':label.get('subjectivity',''),
                'statement':f"This review has a {sentiment.lower()} annotation for {NAMES[aspect]}.",
                'original_support_quote':label.get('evidence',''),'support_span':span,
                'condition_candidate_ids':linked,
                'label_source':'source_human_reviewed_aspect_annotation',
                'fact_source':'deterministic_label_conversion','new_fact_human_reviewed':False,
                'annotation_level':'review_aspect_sentiment_not_full_proposition'}
            local.append(fact);facts.append(fact)
        passages.append({'sample_id':r['sample_id'],'chunk_id':r['chunk_id'],'document_id':r['document_id'],
            'review_id':r['review_id'],'game_name':r['game_name'],'appid':r['appid'],'text':r['text'],
            'source_status':r['status'],'human_annotation':r['human_annotation'],
            'fact_ids':[f['fact_id'] for f in local],
            'condition_candidate_ids':[c['condition_id'] for c in candidates]})
        conditions.extend(candidates)
    if len({f['fact_id'] for f in facts})!=len(facts):raise ValueError('Fact ID collision')
    by_chunk=collections.defaultdict(list)
    for f in facts:by_chunk[f['chunk_id']].append(f)
    condition_map={c['condition_id']:c for c in conditions}
    query_map={q['query_id']:q for q in queries};passage_map={p['chunk_id']:p for p in passages}
    query_annotations=[];contrasts=[]
    for case in generation_cases:
        q=query_map[case['query_id']]
        context_ids=[h['chunk_id'] for h in case['results']]
        for h in case['results']:
            if passage_map[h['chunk_id']]['text']!=h['text']:raise ValueError('Context/source text mismatch')
        pool=[f for cid in context_ids for f in by_chunk[cid]]
        reference=[f for f in pool if fact_is_target(f,q)]
        units=collections.defaultdict(list)
        for f in reference:units[(f['game_name'],f['aspect'],f['sentiment'])].append(f['fact_id'])
        coverage=[{'unit_id':stable_id('U',[case['id'],*key]),'game_name':key[0],
                   'aspect':key[1],'sentiment':key[2],'supporting_fact_ids':ids,
                   'annotation_source':'target_rule_and_human_aspect_labels','human_salience_reviewed':False}
                  for key,ids in sorted(units.items())]
        polar=collections.defaultdict(lambda:collections.defaultdict(list))
        for f in pool:
            if (not q['target_games'] or f['game_name'] in q['target_games']) and f['aspect'] in q['target_aspects']:
                polar[(f['game_name'],f['aspect'])][f['sentiment']].append(f['fact_id'])
        local_contrasts=[]
        for key,sides in sorted(polar.items()):
            if not sides['Positive'] or not sides['Negative']:continue
            group={'contrast_id':stable_id('D',[case['id'],*key]),'query_id':case['id'],
                   'game_name':key[0],'aspect':key[1],'positive_fact_ids':sides['Positive'],
                   'negative_fact_ids':sides['Negative'],'mixed_fact_ids':sides['Mixed'],
                   'required_for_query':q['target_sentiment']=='any',
                   'annotation_source':'opposite_human_polarity_labels','human_reviewed':False,
                   'relation':'different_review_experiences_not_verified_logical_contradiction'}
            contrasts.append(group);local_contrasts.append(group)
        candidate_ids=sorted({c for f in reference for c in f['condition_candidate_ids']})
        query_annotations.append({'id':case['id'],'query_id':case['query_id'],'query':case['query'],
            'query_type':case['query_type'],'retriever':case['retriever'],
            'target_games':q['target_games'],'target_aspects':q['target_aspects'],
            'target_sentiment':q['target_sentiment'],'context_chunk_ids':context_ids,
            'context_fact_ids':[f['fact_id'] for f in pool],
            'reference_fact_ids':[f['fact_id'] for f in reference],'coverage_units':coverage,
            'condition_candidate_ids':candidate_ids,'contrast_groups':local_contrasts,
            'reference_source':'deterministic_target_matching_not_manual_summary_keypoints',
            'human_summary_reference':None,'human_keypoint_reviewed':False,
            'evaluation_only':True})
    write_jsonl(out/'passages.jsonl',passages);write_jsonl(out/'facts.jsonl',facts)
    write_jsonl(out/'conditions_candidates.jsonl',conditions)
    write_jsonl(out/'query_annotations.jsonl',query_annotations)
    write_jsonl(out/'contrast_candidates.jsonl',contrasts)
    with (out/'fact_review.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file,lineterminator='\n');writer.writerow(['fact_id','chunk_id','game','aspect','sentiment','subjectivity','original_quote','alignment','reviewer','decision','notes'])
        for f in facts:writer.writerow([f['fact_id'],f['chunk_id'],f['game_name'],f['aspect'],f['sentiment'],f['subjectivity'],f['original_support_quote'].replace('\r','\\r').replace('\n','\\n'),f['support_span']['status'],'','',''])
    with (out/'condition_review.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file,lineterminator='\n');writer.writerow(['condition_id','chunk_id','kind','mention','context','proposed_fact_ids','reviewer','is_real_condition','is_important','correct_fact_ids','notes'])
        for c in conditions:writer.writerow([c['condition_id'],c['chunk_id'],c['kind'],c['mention'],c['surrounding_text'],json.dumps(c['applies_to_fact_ids']),'','','','',''])
    with (out/'contrast_review.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file,lineterminator='\n');writer.writerow(['contrast_id','query_id','game','aspect','positive_fact_ids','negative_fact_ids','required_for_query','reviewer','decision','notes'])
        for d in contrasts:writer.writerow([d['contrast_id'],d['query_id'],d['game_name'],d['aspect'],json.dumps(d['positive_fact_ids']),json.dumps(d['negative_fact_ids']),d['required_for_query'],'','',''])
    stats={'passages':len(passages),'facts':len(facts),'queries':len(query_annotations),
        'source_aspect_pairs':sum(len(p['human_annotation']) for p in passages),
        'sentiment_counts':dict(collections.Counter(f['sentiment'] for f in facts)),
        'aspect_counts':dict(collections.Counter(f['aspect'] for f in facts)),
        'quote_alignment':dict(collections.Counter(f['support_span']['status'] for f in facts)),
        'condition_candidates':len(conditions),'condition_candidates_linked_to_fact':sum(bool(c['applies_to_fact_ids']) for c in conditions),
        'contrast_candidates':len(contrasts),'required_contrast_groups':sum(d['required_for_query'] for d in contrasts),
        'queries_with_required_contrasts':len({d['query_id'] for d in contrasts if d['required_for_query']}),
        'coverage_units':sum(len(q['coverage_units']) for q in query_annotations),
        'queries_without_reference_facts':[q['id'] for q in query_annotations if not q['reference_fact_ids']],
        'new_labels_human_reviewed':False}
    (out/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',required=True);parser.add_argument('--queries',required=True)
    parser.add_argument('--generation-cases',default='experiments/steam_generation_eval.jsonl')
    parser.add_argument('--output',default='data/generation_annotations/v1')
    args=parser.parse_args()
    stats=build(json.loads(Path(args.annotations).read_text()),read_jsonl(args.queries),read_jsonl(args.generation_cases),args.output)
    manifest={'source_repository':'GSrihariS/RAG_ASPA_Game_Retrieval','source_commit':PIN,
        'source_files':{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.annotations,args.queries,args.generation_cases)},
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'procedure':'deterministic_conversion_and_regex_candidates_v1','uses_llm_api':False,
        'note':'Source aspect labels are human-reviewed; derived summary salience, conditions and contrasts are not human GT.'}
    (Path(args.output)/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
