"""Select the latest available review snapshot and explicit update block."""
from collections import defaultdict
from datetime import datetime, timezone
import math
import re


UPDATE_MARKER = re.compile(
    r'^[ \t]*(?:[\[(][ \t]*)?(?:edit(?:ed)?|update(?:d)?)\b'
    r'(?:[ \t]+(?:edit(?:ed)?|update(?:d)?)\b)*'
    r'(?:[ \t]+(?:[#\d][^:\n\]\)]{0,38}|\([^\n)]{1,38}\)|\[[^\n\]]{1,38}\]))?'
    r'[ \t]*(?:[\])][ \t]*)?[:：][ \t]*', re.I | re.M)
ADMIN_UPDATE = re.compile(
    r'(?:(?:fixed?|corrected|changed|clarified|adjusted)\s+)?(?:a\s+)?'
    r'(?:typos?|spelling|grammar|punctuation|formatting|phrasing|wording)(?:\s+only)?[.!\s]*', re.I)
DATE = re.compile(r'(?<!\d)(?P<year>20\d{2})[-/](?P<month>0?[1-9]|1[0-2])'
                  r'(?:[-/](?P<day>0?[1-9]|[12]\d|3[01]))?(?!\d)'
                  r'|(?<!\d)(?P<month_first>0?[1-9]|1[0-2])/(?P<year_last>20\d{2})(?!\d)')


def timestamp(value):
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)) and math.isfinite(value):
            return datetime.fromtimestamp(value, timezone.utc)
        if isinstance(value, str) and value.strip():
            parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError, OSError):
        pass
    return None


def review_times(meta):
    created = timestamp(meta.get('created_at_utc')) or timestamp(meta.get('timestamp_created'))
    updated = timestamp(meta.get('updated_at_utc')) or timestamp(meta.get('timestamp_updated'))
    return created, updated


def context_time(as_of=None):
    """Capture one UTC clock value per request; allow a fixed clock for replay."""
    if as_of is None:
        return datetime.now(timezone.utc)
    if isinstance(as_of, datetime):
        return (as_of.replace(tzinfo=timezone.utc) if as_of.tzinfo is None
                else as_of.astimezone(timezone.utc))
    parsed = timestamp(as_of)
    if parsed is None:
        raise ValueError('as_of must be a valid ISO timestamp, Unix timestamp, or datetime')
    return parsed


def review_signals(meta, as_of):
    """Whole-review helpfulness per day since first publication, not last edit."""
    signals = {}
    votes = meta.get('helpful_votes')
    if votes is None:
        votes = meta.get('votes_up')
    if isinstance(votes, str) and votes.isascii() and votes.isdigit():
        try:
            votes = int(votes)
        except ValueError:
            votes = None
    if type(votes) is int and votes >= 0:
        signals['helpful_votes'] = votes
    created, _ = review_times(meta)
    if created is not None and created <= as_of:
        age_days = (as_of - created).total_seconds() / 86400
        signals['review_age_days'] = round(age_days, 6)
        if 'helpful_votes' in signals:
            signals['helpful_votes_per_day'] = round(votes / max(age_days, 1.0), 6)
    return signals


def review_key(hit):
    meta = hit['metadata']
    steam = re.match(r'^(steam:[^:]+:[^:]+):chunk:\d+$', hit['chunk_id'])
    if steam:
        return ('steam', steam.group(1))
    if meta.get('review_id') is not None:
        return ('review', str(meta.get('appid') or meta.get('game_name') or ''), str(meta['review_id']))
    if meta.get('document_id'):
        return ('document', str(meta['document_id']))
    return ('chunk', hit['chunk_id'])


def update_date(text):
    # Use a date in the marker itself or following an explicit "as of/on" cue;
    # a historical date mentioned in the body does not timestamp the update.
    marker = UPDATE_MARKER.match(text)
    header = text[:marker.end()] if marker else ''
    match = DATE.search(header)
    if match is None:
        cue = re.search(r'\b(?:as\s+of|as\s+at|updated?\s+on)\s+', text[:400], re.I)
        if cue:
            match = DATE.match(text, cue.end())
    if match is None:
        return None
    try:
        return datetime(int(match['year'] or match['year_last']),
                        int(match['month'] or match['month_first']),
                        int(match['day'] or 1), tzinfo=timezone.utc)
    except ValueError:
        return None


def chunk_order(hit, fallback):
    meta = hit['metadata']
    value = meta.get('chunk_index')
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    match = re.search(r':chunk:(\d+)$', hit['chunk_id'])
    return int(match.group(1)) if match else fallback


def latest_review_hits(hits):
    """Return copies; preserve source IDs, originals and the rank of retained hits.

    Different chunks from one snapshot are retained. When a visible update spans
    several chunks, retain its continuation and exclude earlier/later blocks.
    """
    if not isinstance(hits, list):
        raise ValueError('hits must be a ranked list')
    groups = defaultdict(list)
    for rank, raw in enumerate(hits):
        if (not isinstance(raw, dict) or not isinstance(raw.get('chunk_id'), str)
                or not raw['chunk_id'].strip() or not isinstance(raw.get('text'), str)):
            raise ValueError('Each hit needs original nonempty chunk_id and string text')
        meta = raw.get('metadata') or {}
        if not isinstance(meta, dict):
            raise ValueError('metadata must be an object or null')
        hit = dict(raw, metadata=dict(meta))
        created, updated = review_times(meta)
        groups[review_key(hit)].append((rank, hit, updated or created))

    selected = []
    for group in groups.values():
        dates = [t for _, _, t in group if t is not None]
        if dates:
            newest = max(dates)
            group = [item for item in group if item[2] == newest]
        offsets = all(isinstance(h['metadata'].get('char_start'), int)
                      and not isinstance(h['metadata']['char_start'], bool)
                      and h['metadata']['char_start'] >= 0 for _, h, _ in group)
        markers = {}
        positions = {}
        for rank, hit, _ in group:
            base = hit['metadata']['char_start'] if offsets else chunk_order(hit, rank)
            positions[rank] = base
            for match in UPDATE_MARKER.finditer(hit['text']):
                position = (base + match.start(), 0) if offsets else (base, match.start())
                markers.setdefault(position, (hit['text'], match))
        marker_positions = sorted(markers)

        def bounds(rank, hit, start, end):
            base = positions[rank]
            if offsets:
                return (max(0, start[0] - base),
                        min(len(hit['text']), end[0] - base) if end else len(hit['text']))
            if base < start[0] or (end is not None and base > end[0]):
                return 0, 0
            return (start[1] if base == start[0] else 0,
                    end[1] if end is not None and base == end[0] else len(hit['text']))

        meaningful = []
        for i, position in enumerate(marker_positions):
            text, match = markers[position]
            end = marker_positions[i + 1] if i + 1 < len(marker_positions) else None
            body_start = ((position[0] + match.end() - match.start(), 0) if offsets
                          else (position[0], match.end()))
            # An update heading may be at the end of one chunk, with its body
            # in the next. Gather visible continuation only for classification;
            # the model still receives separately cited original substrings.
            parts, consumed = [], -1
            for rank, hit, _ in sorted(group, key=lambda item: positions[item[0]]):
                lo, hi = bounds(rank, hit, body_start, end)
                if offsets:
                    lo = max(lo, consumed - positions[rank])
                    consumed = max(consumed, positions[rank] + hi)
                if lo < hi:
                    parts.append(hit['text'][lo:hi])
            body = '\n'.join(parts).strip().rstrip(')]').strip()
            if body and not ADMIN_UPDATE.fullmatch(body):
                meaningful.append((position, update_date(text[match.start():match.end()] + body)))
        start = end = None
        if meaningful:
            # If every meaningful update is dated, timestamps also handle
            # reverse-chronological blocks. Otherwise retain the last explicit one.
            start, _ = max(meaningful, key=lambda item: (item[1], item[0])) if all(
                date is not None for _, date in meaningful) else meaningful[-1]
            end = next((p for p in marker_positions if p > start), None)
        for rank, hit, _ in group:
            text, base = hit['text'], positions[rank]
            lo, hi = 0, len(text)
            if start is not None:
                lo, hi = bounds(rank, hit, start, end)
            if lo >= hi or not text[lo:hi].strip():
                continue
            hit['text'] = text[lo:hi]
            hit['metadata']['evidence_scope'] = ('latest_explicit_update' if start is not None
                                                  else 'latest_available_snapshot')
            hit['metadata']['selected_text_start'] = lo
            hit['metadata']['selected_text_end'] = hi
            selected.append((rank, hit))
    return [hit for _, hit in sorted(selected, key=lambda item: item[0])]
