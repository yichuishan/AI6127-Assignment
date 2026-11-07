"""Deterministic, query-aware extractive compression; no labels, API or dependencies."""
from collections import Counter
import copy
import math
import re

VERSION = 'tfidf_sentence_windows_v1'
DEFAULT_RATIO = 0.6
STOP_WORDS = set('a an the and or but of to in on for with is are was were be been it its this that these those do does did how what which why when where who players player reviewers review reviews think about game games have has had as at by from you your they their can would still'.split())
GAP = '\n[…]\n'


def terms(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOP_WORDS]


def sentence_spans(text):
    """Offsets preserve original punctuation, case, conditions and negation."""
    spans, start = [], 0
    for boundary in re.finditer(r'(?<=[.!?])\s+|\n+', text):
        end = boundary.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = boundary.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def compress_context(question, context, ratio=DEFAULT_RATIO):
    if type(ratio) not in (float, int) or not math.isfinite(ratio) or not .2 <= ratio <= 1:
        raise ValueError('Compression ratio must be between 0.2 and 1')
    original_chars = sum(len(e['text']) for e in context)
    budget = max(1, math.floor(original_chars * ratio))
    spans = [sentence_spans(e['text']) for e in context]
    units = [(i, j, terms(e['text'][a:b])) for i, e in enumerate(context)
             for j, (a, b) in enumerate(spans[i])]
    # Fit TF-IDF to this query's evidence only; no reference answers or labels.
    df = Counter(t for _, _, ts in units for t in set(ts))
    idf = {t: math.log((1 + len(units)) / (1 + n)) + 1 for t, n in df.items()}

    def vector(ts):
        v = {t: (1 + math.log(n)) * idf.get(t, math.log(1 + len(units)) + 1)
             for t, n in Counter(ts).items()}
        norm = math.sqrt(sum(x*x for x in v.values()))
        return {t:x/norm for t,x in v.items()} if norm else {}

    q = vector(terms(question))
    vectors = [vector(ts) for _, _, ts in units]
    relevance = [sum(value*q.get(t, 0) for t,value in v.items()) for v in vectors]
    selected = [set() for _ in context]
    chosen_vectors = []

    def ranges(i, indices):
        groups = []
        for j in sorted(indices):
            a,b = spans[i][j]
            if groups and j == groups[-1][2] + 1:
                groups[-1] = (groups[-1][0], b, j)
            else:
                groups.append((a,b,j))
        return [(a,b) for a,b,_ in groups]

    def size(candidate):
        return sum(sum(b-a for a,b in ranges(i, js)) + max(0,len(ranges(i,js))-1)*len(GAP)
                   for i,js in enumerate(candidate))

    if ratio == 1:
        # Identity mode is useful for development and preserves whitespace exactly.
        selected = [set(range(len(ss))) for ss in spans]
    else:
        remaining = set(range(len(units)))
        while remaining:
            def priority(k):
                redundancy = max((sum(x*vectors[k].get(t,0) for t,x in v.items())
                                  for v in chosen_vectors), default=0)
                return (.8*relevance[k] - .2*redundancy, relevance[k], -k)
            k = max(remaining, key=priority)
            remaining.remove(k)
            i,j,_ = units[k]
            if j in selected[i]:
                continue
            # Keep a complete three-sentence window when it fits. At the budget
            # boundary keep whole sentences, never truncate a sentence mid-claim.
            options = [set(range(max(0,j-1), min(len(spans[i]),j+2))), {j}]
            for window in options:
                candidate = [set(js) for js in selected]
                candidate[i].update(window)
                if size(candidate) <= budget:
                    selected = candidate
                    chosen_vectors.append(vectors[k])
                    break
        if not any(selected) and units:
            # An indivisible long sentence may exceed the requested budget.
            k = max(range(len(units)), key=lambda k:(relevance[k], -k))
            i,j,_ = units[k]
            selected[i].add(j)

    compressed, kept = [], []
    for i,e in enumerate(context):
        rr = [(0,len(e['text']))] if ratio == 1 else ranges(i,selected[i])
        if not rr:
            continue
        entry = copy.deepcopy(e)
        entry['text'] = GAP.join(e['text'][a:b] for a,b in rr)
        compressed.append(entry)
        kept.append({'chunk_id':e['chunk_id'], 'spans':[[a,b] for a,b in rr]})
    chars = sum(len(e['text']) for e in compressed)
    return compressed, {'version':VERSION,'requested_ratio':ratio,'target_text_chars':budget,
        'original_text_chars':original_chars,'retained_text_chars':chars,
        'actual_text_ratio':chars/original_chars if original_chars else 0,
        'original_chunks':len(context),'retained_chunks':len(compressed),
        'over_budget':chars>budget,'neighbor_sentences':1,'mmr_relevance_weight':.8,
        'fallback':'whole highest-ranked sentence if no sentence fits', 'retained_spans':kept}
