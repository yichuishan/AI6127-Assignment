"""Independent evidence-grounded generation. Python 3.10+, no runtime dependencies."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import ssl
from urllib.error import HTTPError, URLError
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

PROMPTS = Path(__file__).with_name('prompts')
APPROACHES = ('baseline', 'improved', 'verify_delete', 'verify_revise')
VERDICTS = ('supported', 'partially_supported', 'contradicted', 'unsupported')
ABSTENTION = 'The retrieved reviews do not provide sufficient evidence to answer this question.'


ERROR_MESSAGES = {
    'generation_error': '生成或评估失败，请查看错误阶段与配置。',
    'missing_key': '缺少 API Key。',
    'http_400': 'API 拒绝请求（400）：检查模型名称、参数和 JSON 模式支持。',
    'http_401': '认证失败（401）：检查密钥及其所属服务。',
    'http_402': '账户余额不足或计费不可用（402）。',
    'http_403': '服务拒绝访问（403）：检查账户权限或网络限制。',
    'http_404': '接口或模型不存在（404）：检查 API 基础地址和模型名。',
    'http_422': '请求参数不被接受（422）：检查模型和参数兼容性。',
    'http_429': '请求被限流或配额不足（429），本程序不会自动付费重试。',
    'http_5xx': '模型服务暂时异常（5xx），请稍后重试。',
    'http_error': '模型 API 返回 HTTP 错误。',
    'timeout': '模型请求超时；思考模式可关闭或增加请求超时。',
    'tls_error': 'TLS 证书验证失败，请检查本机证书和代理配置。',
    'network_error': '无法连接模型 API，请检查地址、网络和代理。',
    'response_schema': 'API 返回结构不是预期的 Chat Completions 格式。',
    'output_truncated': '输出达到 token 上限而截断；增加输出预算，或关闭 DeepSeek 思考模式。',
    'empty_content': 'API 返回空白正文；可能是 JSON 模式输出空白或思考耗尽预算。',
    'invalid_json': '模型正文不是有效 JSON，无法解析。',
    'invalid_fact_ids': '结构化事实选择格式错误：每条观点必须包含 fact_ids 字符串列表。',
    'invalid_claims': '模型未返回符合要求的 claims 列表，或观点数超过限制。',
    'invalid_verdict': '验证器返回的判定格式不符合要求。',
    'invalid_revision': '修订器未返回要求的 claim 字段。',
    'invalid_judge': '最终评估模型返回的评分格式不符合要求。',
}


class GenerationError(RuntimeError):
    """Safe structured errors; never retain raw provider bodies or credentials."""
    def __init__(self, message, *, code='generation_error', http_status=None, stage=None):
        super().__init__(message)
        self.code = code if code in ERROR_MESSAGES else 'generation_error'
        self.http_status = http_status
        self.stage = stage


def safe_error(error):
    if not isinstance(error, GenerationError):
        return {'error_type': type(error).__name__, 'error_message': '运行失败，请检查输入、依赖和配置。'}
    result = {'error_type': type(error).__name__, 'error_code': error.code,
              'error_message': ERROR_MESSAGES[error.code]}
    if error.http_status is not None:
        result['http_status'] = error.http_status
    if error.stage in ('generate', 'verify', 'revise', 'reverify', 'judge'):
        result['error_stage'] = error.stage
    if error.code == 'invalid_judge' and hasattr(error,'validation_issues'):
        result['validation_issues'] = error.validation_issues
    return result


def load_retrieval(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('query'), str) or not isinstance(data.get('results'), list):
        raise ValueError('Expected Retrieval export with query (string) and results (list)')
    return data['query'], data['results']


def build_context(hits, max_context_chars=24000, max_chunk_chars=4000):
    """Keep ranking, deduplicate IDs and bound serialized context without mutation."""
    if not isinstance(hits, list):
        raise ValueError('hits must be a ranked list, not raw sparse/dense output')
    if max_context_chars < 100 or max_chunk_chars < 1:
        raise ValueError('Invalid context limits')
    selected, seen = [], set()
    for hit in hits:
        if not isinstance(hit, dict):
            raise ValueError('Each hit must be an object')
        cid, text = hit.get('chunk_id'), hit.get('text')
        if not isinstance(cid, str) or not cid.strip() or not isinstance(text, str):
            raise ValueError('Each hit needs original nonempty chunk_id and string text')
        if cid in seen or not text.strip():
            continue
        seen.add(cid)
        meta = hit.get('metadata') or {}
        if not isinstance(meta, dict):
            raise ValueError('metadata must be an object or null')
        url = meta.get('source_url')
        url = url if isinstance(url, str) and urlparse(url).scheme in ('http', 'https') and urlparse(url).netloc else None
        entry = {'chunk_id': cid, 'text': text[:max_chunk_chars],
                 'game_name': str(meta.get('game_name') or '')[:200], 'source_url': url}
        if len(json.dumps(selected + [entry], ensure_ascii=False)) > max_context_chars:
            break
        selected.append(entry)
    return selected


class OpenAICompatibleClient:
    """Chat Completions transport using inherited proxy and TLS trust."""
    def __init__(self, base_url=None, api_key=None, timeout=180, max_tokens=8192, thinking='auto'):
        self.base_url = (base_url or os.getenv('GENERATION_BASE_URL', 'https://api.openai.com/v1')).rstrip('/')
        self.api_key = api_key or os.getenv('GENERATION_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.timeout = timeout
        if type(max_tokens) is not int or not 256 <= max_tokens <= 131072:
            raise ValueError('max_tokens must be 256..131072')
        if thinking not in ('auto', 'disabled', 'enabled'):
            raise ValueError('Invalid thinking mode')
        self.max_tokens, self.thinking = max_tokens, thinking
        parsed = urlparse(self.base_url)
        if parsed.scheme not in ('https', 'http') or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError('Invalid API base URL')
        if parsed.scheme == 'http' and parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('Use HTTPS for remote endpoints')

    def complete(self, messages, model, temperature=0):
        if not self.api_key and urlparse(self.base_url).hostname not in ('localhost', '127.0.0.1', '::1'):
            raise GenerationError('Set GENERATION_API_KEY or OPENAI_API_KEY for hosted inference', code='missing_key')
        payload = {'model': model, 'messages': messages, 'temperature': temperature,
                   'max_tokens': self.max_tokens, 'response_format': {'type': 'json_object'}}
        thinking = self.thinking
        host = urlparse(self.base_url).hostname or ''
        aliyun_qwen = (host == 'aliyuncs.com' or host.endswith('.aliyuncs.com')) and model.lower().startswith('qwen')
        if thinking == 'auto':
            if host == 'api.deepseek.com' and (model.startswith('deepseek-v4') or model == 'deepseek-flash'):
                thinking = 'disabled'
            elif aliyun_qwen and model.lower().startswith('qwen3'):
                thinking = 'disabled'
        if thinking != 'auto':
            if aliyun_qwen:
                payload['enable_thinking'] = thinking == 'enabled'
            else:
                payload['thinking'] = {'type': thinking}
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = 'Bearer ' + self.api_key
        request = Request(self.base_url + '/chat/completions', json.dumps(payload).encode(), headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except HTTPError as error:
            status = error.code
            code = 'http_' + str(status)
            if code not in ERROR_MESSAGES:
                code = 'http_5xx' if 500 <= status < 600 else 'http_error'
            raise GenerationError(ERROR_MESSAGES[code], code=code, http_status=status) from None
        except (TimeoutError, socket.timeout):
            raise GenerationError(ERROR_MESSAGES['timeout'], code='timeout') from None
        except URLError as error:
            code = 'tls_error' if isinstance(error.reason, ssl.SSLCertVerificationError) else 'timeout' if isinstance(error.reason, (TimeoutError, socket.timeout)) else 'network_error'
            raise GenerationError(ERROR_MESSAGES[code], code=code) from None
        except Exception:
            raise GenerationError(ERROR_MESSAGES['response_schema'], code='response_schema') from None
        try:
            choice = data['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise GenerationError(ERROR_MESSAGES['output_truncated'], code='output_truncated')
            content = choice['message'].get('content')
            if content is None or (isinstance(content, str) and not content.strip()):
                raise GenerationError(ERROR_MESSAGES['empty_content'], code='empty_content')
            if not isinstance(content, str):
                raise ValueError()
            return {'content': content, 'usage': data.get('usage', {})}
        except GenerationError:
            raise
        except (KeyError, IndexError, TypeError, ValueError):
            raise GenerationError(ERROR_MESSAGES['response_schema'], code='response_schema') from None



def validate_claim(claim, sources, require_quotes=True, require_fact_ids=False):
    """Normalize a claim or return None. Provenance checks are not semantic NLI."""
    if not isinstance(claim, dict):
        return None
    text, ids = claim.get('text'), claim.get('chunk_ids')
    if not (isinstance(text, str) and text.strip() and isinstance(ids, list) and ids
            and all(isinstance(cid, str) and cid in sources for cid in ids)):
        return None
    evidence = claim.get('evidence', [])
    if require_quotes:
        if not isinstance(evidence, list) or not evidence:
            return None
        checked = set()
        for item in evidence:
            if not isinstance(item, dict):
                return None
            cid, quote = item.get('chunk_id'), item.get('quote')
            if (not isinstance(cid, str) or cid not in ids or not isinstance(quote, str)
                    or not quote.strip() or quote not in sources[cid]['text']):
                return None
            checked.add(cid)
        if checked != set(ids):
            return None
    normalized = {'text': text.strip(), 'chunk_ids': list(dict.fromkeys(ids)),
                  'evidence': evidence if require_quotes else []}
    if require_fact_ids:
        values = claim.get('fact_ids')
        if not isinstance(values,list) or any(not isinstance(x,str) or not x.strip() for x in values):
            return None
        # Preserve incorrect IDs as predictions; the deterministic scorer penalizes them.
        normalized['fact_ids'] = list(dict.fromkeys(values))
    return normalized


def generate_summary(query, hits, *, approach='verify_revise', model=None, client=None,
                     max_context_chars=24000, max_chunk_chars=4000,
                     input_cost_per_million=None, output_cost_per_million=None, max_claims=12, fact_candidates=None):
    """Generate, verify/delete or verify/revise/reverify using the same model/client."""
    started = time.perf_counter()
    if not isinstance(query, str) or not query.strip() or len(query) > 8000:
        raise ValueError('query must be nonempty and at most 8000 characters')
    if approach not in APPROACHES:
        raise ValueError('Unknown approach: ' + str(approach))
    if type(max_claims) is not int or not 1 <= max_claims <= 50:
        raise ValueError('max_claims must be between 1 and 50')
    for price in (input_cost_per_million, output_cost_per_million):
        if price is not None and (not math.isfinite(price) or price < 0):
            raise ValueError('Token prices must be finite and nonnegative')
    context = build_context(hits, max_context_chars, max_chunk_chars)
    sources = {x['chunk_id']: x for x in context}
    fact_mode = fact_candidates is not None
    if fact_mode:
        if not isinstance(fact_candidates,list) or any(not isinstance(c,dict) or set(c)!={'fact_id','chunk_id','aspect'} or c['chunk_id'] not in sources for c in fact_candidates):
            raise ValueError('Fact candidates must be label-free catalog entries for the visible context')
    model = model or os.getenv('GENERATION_MODEL', 'gpt-4o-mini')
    semantic = approach in ('verify_delete', 'verify_revise')
    result = {'query': query, 'summary': ABSTENTION, 'citations': [], 'claims': [],
              'model': model, 'approach': approach, 'status': 'no_evidence', 'usage': {},
              'estimated_cost_usd': None, 'context_chunk_ids': list(sources),
              'stage_calls': [], 'claim_audit': [],
              'validation': {'rejected_claims': 0, 'quote_check': approach != 'baseline',
                             'semantic_check': semantic, 'generated_claims': 0,
                             'provenance_valid_claims': 0, 'retained_original_claims': 0,
                             'revised_claims': 0, 'semantic_discarded_claims': 0}}
    if fact_mode: result['selected_fact_ids'] = []
    transport = client

    def invoke(stage, prompt, payload, claim_id=None):
        nonlocal transport
        if transport is None:
            transport = OpenAICompatibleClient()
        system = (PROMPTS / (prompt + '.txt')).read_text(encoding='utf-8')
        if fact_mode:
            system += '\n' + (PROMPTS / 'fact_selection.txt').read_text(encoding='utf-8')
            payload = dict(payload, fact_candidates=fact_candidates)
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
        call_started = time.perf_counter()
        try:
            response = transport.complete(messages, model=model, temperature=0)
        except GenerationError as error:
            error.stage = stage
            raise
        result['stage_calls'].append({'stage': stage, 'claim_id': claim_id, 'model': model,
                                      'latency_seconds': round(time.perf_counter() - call_started, 6),
                                      'usage': response.get('usage') or {}})
        try:
            data = json.loads(response['content'])
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except (ValueError, TypeError, KeyError):
            raise GenerationError('Invalid JSON object from ' + stage, code='invalid_json', stage=stage) from None

    def verify(claim, claim_id, stage):
        data = invoke(stage, 'verify', {'question': query, 'claim': claim, 'evidence': context}, claim_id)
        if (data.get('verdict') not in VERDICTS
                or not isinstance(data.get('reason'), str) or not data['reason'].strip()):
            raise GenerationError('Invalid semantic verification result', code='invalid_verdict', stage=stage)
        return {'verdict': data['verdict'], 'reason': data['reason']}

    if context:
        output = invoke('generate', 'baseline' if approach == 'baseline' else 'improved',
                        {'question': query, 'evidence': context, 'max_claims': max_claims})
        claims = output.get('claims')
        if not isinstance(claims, list) or len(claims) > max_claims:
            raise GenerationError('Model must return a claims list within max_claims', code='invalid_claims', stage='generate')
        if fact_mode and any(not isinstance(c,dict) or not isinstance(c.get('fact_ids'),list) or any(not isinstance(x,str) or not x.strip() for x in c['fact_ids']) for c in claims):
            raise GenerationError('Missing or invalid fact_ids',code='invalid_fact_ids',stage='generate')
        stats = result['validation']
        stats['generated_claims'] = len(claims)
        accepted = []
        for index, raw in enumerate(claims):
            claim_id = 'c' + str(index + 1)
            claim = validate_claim(raw, sources, approach != 'baseline', require_fact_ids=fact_mode)
            audit = {'claim_id': claim_id, 'original': raw, 'outcome': 'provenance_rejected'}
            result['claim_audit'].append(audit)
            if claim is None:
                stats['rejected_claims'] += 1
                continue
            stats['provenance_valid_claims'] += 1
            if not semantic:
                accepted.append(claim)
                audit['outcome'] = 'kept_without_semantic_check'
                stats['retained_original_claims'] += 1
                continue
            first = verify(claim, claim_id, 'verify')
            audit['verification'] = first
            if first['verdict'] == 'supported':
                accepted.append(claim)
                audit['outcome'] = 'kept'
                stats['retained_original_claims'] += 1
                continue
            audit['outcome'] = 'semantic_discarded'
            if approach == 'verify_revise':
                repair = invoke('revise', 'revise', {'question': query, 'claim': claim,
                                'verification': first, 'evidence': context}, claim_id)
                if 'claim' not in repair:
                    raise GenerationError('Revision must contain claim (object or null)', code='invalid_revision', stage='revise')
                audit['revision_candidate'] = repair['claim']
                revised = validate_claim(repair['claim'], sources, require_fact_ids=fact_mode)
                if revised is not None:
                    second = verify(revised, claim_id, 'reverify')
                    audit['revision_verification'] = second
                    if second['verdict'] == 'supported':
                        accepted.append(revised)
                        audit['outcome'] = 'revised'
                        stats['revised_claims'] += 1
                        continue
                audit['outcome'] = 'revision_abstained' if repair['claim'] is None else 'revision_rejected'
            stats['semantic_discarded_claims'] += 1
        if fact_mode:
            result['selected_fact_ids'] = list(dict.fromkeys(fid for c in accepted for fid in c['fact_ids']))
        cited = list(dict.fromkeys(cid for c in accepted for cid in c['chunk_ids']))
        result.update(claims=accepted,
                      citations=[{'chunk_id': cid, 'source_url': sources[cid]['source_url']} for cid in cited],
                      summary=' '.join(c['text'] + ' ' + ' '.join('[' + cid + ']' for cid in c['chunk_ids']) for c in accepted) if accepted else ABSTENTION,
                      status='ok' if accepted else 'abstained')
        calls = result['stage_calls']
        if all(all(type(c['usage'].get(k)) is int and c['usage'][k] >= 0
                   for k in ('prompt_tokens', 'completion_tokens')) for c in calls):
            result['usage'] = {k: sum(c['usage'][k] for c in calls) for k in ('prompt_tokens', 'completion_tokens')}
            if input_cost_per_million is not None and output_cost_per_million is not None:
                result['estimated_cost_usd'] = (result['usage']['prompt_tokens'] * input_cost_per_million
                        + result['usage']['completion_tokens'] * output_cost_per_million) / 1e6
    result['latency_seconds'] = round(time.perf_counter() - started, 6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('retrieval_json')
    parser.add_argument('--approach', choices=APPROACHES, default='verify_revise')
    parser.add_argument('--output', default='generation_results.json')
    parser.add_argument('--max-claims', type=int, default=12)
    args = parser.parse_args()
    query, hits = load_retrieval(args.retrieval_json)
    result = generate_summary(query, hits, approach=args.approach, max_claims=args.max_claims)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
