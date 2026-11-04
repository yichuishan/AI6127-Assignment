"""Private experiment UI: python experiment_web.py --port 8080. No external dependencies."""
import argparse
import io
import ipaddress
import json
import math
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlparse
import zipfile

from rag_generator import APPROACHES, OpenAICompatibleClient
from run_generation_experiments import jsonl, parser, run

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / 'experiments/steam_generation_eval.jsonl'


class Cancelled(Exception):
    pass


def endpoint(value):
    value = value.strip().rstrip('/')
    # Accept both the API base URL and the familiar complete Chat Completions URL.
    if value.endswith('/chat/completions'):
        value = value[:-len('/chat/completions')]
    p = urlparse(value)
    if p.scheme != 'https' or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ValueError('API 地址须为 HTTPS，不得含账号、密码、查询参数或片段。')
    try:
        if not ipaddress.ip_address(p.hostname).is_global:
            raise ValueError('请使用公网模型 API 地址。')
    except ValueError as error:
        if p.hostname.lower() in ('localhost', 'metadata.google.internal') or p.hostname.endswith('.local'):
            raise ValueError('请使用公网模型 API 地址。') from error
        # Distinguish a DNS hostname from a rejected numeric private address.
        try:
            ipaddress.ip_address(p.hostname)
        except ValueError:
            pass
        else:
            raise error
    return value


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'整数参数应在 {minimum}–{maximum} 之间。')
    return value


class LiveClient:
    def __init__(self, job, key, base_url, factory):
        self.job = job
        self.transport = factory(base_url=base_url, api_key=key, timeout=60)

    def complete(self, messages, model, temperature):
        job = self.job
        if job.stop.is_set():
            raise Cancelled()
        system = messages[0]['content']
        stage = ('最终评估' if 'Evaluate an answer' in system else
                 '证据核验' if 'evidence verifier' in system else
                 '修订观点' if 'Repair one rejected' in system else '生成草稿')
        with job.lock:
            job.state.update(stage=stage, api_calls=job.state['api_calls'] + 1)
        response = self.transport.complete(messages, model, temperature)
        with job.lock:
            job.state['successful_api_calls'] += 1
        return response


class Job:
    def __init__(self, owner, data, directory, factory):
        self.owner, self.directory = owner, directory
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.state = {'id': directory.name, 'status': 'running', 'completed': 0, 'total': 0,
                      'failed': 0, 'api_calls': 0, 'successful_api_calls': 0,
                      'stage': '准备输入', 'events': [], 'started_at': time.time()}
        self.thread = threading.Thread(target=self.execute, args=(data, factory), daemon=True)

    def execute(self, data, factory):
        try:
            key = data.pop('api_key')
            judge_key = data.pop('judge_api_key', '') or key
            client = LiveClient(self, key, data['base_url'], factory)
            judge = LiveClient(self, judge_key, data.get('judge_base_url') or data['base_url'], factory)
            # Credentials stay only in the two transport objects, never in config/files/logs.
            key = judge_key = None
            dataset = self.directory / 'input.jsonl'
            cases = data.pop('cases')
            dataset.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in cases), encoding='utf-8')
            args = parser().parse_args(['run', str(dataset), '--output', str(self.directory/'results')])
            for name in ('model', 'base_url', 'judge_model', 'judge_base_url', 'approaches',
                         'repeats', 'max_claims', 'reference_metrics', 'input_price', 'output_price',
                         'judge_input_price', 'judge_output_price'):
                if name in data:
                    setattr(args, name, data[name])
            with self.lock:
                self.state['total'] = len(cases) * args.repeats * len(args.approaches)
            def progress(row, completed, total):
                with self.lock:
                    self.state.update(completed=completed, total=total, stage='完成一项实验')
                    self.state['failed'] += row['status'] == 'failed'
                    self.state['events'].append({'case_id': row['case_id'], 'approach': row['approach'],
                        'status': row['status'], 'error_type': row.get('error_type'),
                        'summary': row.get('result', {}).get('summary', '')[:1800]})
                    self.state['events'] = self.state['events'][-30:]
                    if len(self.state['events']) >= 3 and all(e['status'] == 'failed' for e in self.state['events'][-3:]):
                        self.state['stop_reason'] = '连续 3 项实验失败，已自动停止；请检查 API 配置与结果文件。'
                        self.stop.set()
            run(args, client, judge, progress=progress, stop_requested=self.stop.is_set)
            with self.lock:
                self.state.update(status='stopped' if self.stop.is_set() else 'completed', stage='已停止' if self.stop.is_set() else '实验结束')
        except Exception as error:
            with self.lock:
                self.state.update(status='failed', stage='运行失败', error_type=type(error).__name__)
        finally:
            data.clear()
            with self.lock:
                self.state['finished_at'] = time.time()

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))


class Application:
    def __init__(self, run_root=None, factory=OpenAICompatibleClient):
        self.run_root = Path(run_root or ROOT/'experiments/web_runs')
        self.run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.factory, self.jobs, self.sessions, self.lock = factory, {}, set(), threading.Lock()

    def start(self, owner, data):
        key = data.get('api_key', '')
        if not isinstance(key, str) or not key.strip() or len(key) > 4096:
            raise ValueError('请输入 API Key。')
        data['api_key'] = key.strip()
        if not isinstance(data.get('judge_api_key', ''), str) or len(data.get('judge_api_key', '')) > 4096:
            raise ValueError('评估 API Key 格式无效。')
        data['base_url'] = endpoint(data.get('base_url', ''))
        if data.get('judge_base_url'):
            data['judge_base_url'] = endpoint(data['judge_base_url'])
        for name in ('model', 'judge_model'):
            value = data.get(name, '')
            if not isinstance(value, str) or len(value) > 200 or (name == 'model' and not value.strip()):
                raise ValueError('请输入有效模型名称。')
            data[name] = value.strip() or None
        data['repeats'] = integer(data.get('repeats', 1), 1, 5)
        data['max_claims'] = integer(data.get('max_claims', 4), 1, 12)
        limit = integer(data.get('limit', 3), 1, 1000)
        arms = data.get('approaches', list(APPROACHES))
        if not isinstance(arms, list) or not arms or any(a not in APPROACHES for a in arms) or len(set(arms)) != len(arms):
            raise ValueError('请选择至少一种有效实验方法。')
        data['approaches'] = arms
        metrics = data.get('reference_metrics', [])
        if not isinstance(metrics, list) or any(m not in ('rouge', 'bertscore') for m in metrics):
            raise ValueError('指标选择无效。')
        for name in ('input_price', 'output_price', 'judge_input_price', 'judge_output_price'):
            value = data.get(name)
            if value is not None and (type(value) not in (float, int) or not math.isfinite(value) or value < 0):
                raise ValueError('价格应为非负数字或留空。')
        text = data.pop('dataset_jsonl', '')
        cases = [json.loads(line) for line in text.splitlines() if line.strip()] if text else jsonl(DEFAULT_DATASET)
        cases = cases[:limit]
        if not cases or any(not isinstance(c, dict) or not isinstance(c.get('id'), str) or not c['id'].strip() for c in cases):
            raise ValueError('数据集需要非空字符串 id。')
        if len({c['id'] for c in cases}) != len(cases):
            raise ValueError('数据集 id 重复。')
        from rag_generator import build_context
        for c in cases:
            if not isinstance(c.get('query'), str) or not c['query'].strip() or len(c['query']) > 8000:
                raise ValueError('数据集 query 无效。')
            build_context(c['results'])
        if metrics:
            import importlib.util
            for metric, module in [('rouge', 'rouge_score'), ('bertscore', 'bert_score')]:
                if metric in metrics and not importlib.util.find_spec(module):
                    raise ValueError('当前环境未安装所选参考指标依赖，请先安装 requirements-generation-eval.txt。')
            if any(not c.get('reference') for c in cases):
                raise ValueError('所选参考指标需要每条输入具有人工 reference，请上传已标注 JSONL。')
        data['cases'] = cases
        with self.lock:
            if any(j.snapshot()['status'] == 'running' for j in self.jobs.values()):
                raise ValueError('已有实验运行中，请等待或停止后再开始。')
            directory = self.run_root / secrets.token_hex(16)
            directory.mkdir(mode=0o700)
            job = Job(owner, data, directory, self.factory)
            self.jobs[directory.name] = job
            job.thread.start()
        return job


def server(host='127.0.0.1', port=8080, application=None):
    app = application or Application()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log request bodies, credentials, or download tokens.

        def send(self, status, body, kind='application/json', cookie=None, attachment=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'self'")
            if cookie:
                self.send_header('Set-Cookie', f'experiment_session={cookie}; HttpOnly; SameSite=Strict; Path=/')
            if attachment:
                self.send_header('Content-Disposition', f'attachment; filename="{attachment}"')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def owner(self):
            try:
                cookies = SimpleCookie(self.headers.get('Cookie', ''))
                token = cookies['experiment_session'].value
                return token if token in app.sessions else None
            except (KeyError, ValueError):
                return None

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                token = self.owner() or secrets.token_urlsafe(32)
                app.sessions.add(token)
                return self.send(200, (ROOT/'web/index.html').read_bytes(), 'text/html; charset=utf-8', cookie=token)
            if path in ('/app.js', '/style.css'):
                return self.send(200, (ROOT/'web'/path[1:]).read_bytes(), 'text/javascript' if path.endswith('.js') else 'text/css')
            owner = self.owner()
            if not owner:
                return self.send(401, {'error': '请先打开页面。'})
            if path == '/api/info':
                cases = jsonl(DEFAULT_DATASET)
                return self.send(200, {'cases': len(cases), 'references': sum(bool(c.get('reference')) for c in cases),
                    'dataset_status': '已验证网站及标注接口 HTTP 200；1,000 条在线标注与指定 GitHub 版本完全一致（2026-11-04）。',
                    'dataset_access': json.loads((ROOT/'experiments/dataset_access.json').read_text()) if (ROOT/'experiments/dataset_access.json').exists() else None,
                    'jobs': [j.snapshot() for j in app.jobs.values() if j.owner == owner]})
            parts = path.strip('/').split('/')
            if len(parts) == 3 and parts[:2] == ['api', 'jobs']:
                job = app.jobs.get(parts[2])
                if not job or job.owner != owner:
                    return self.send(404, {'error': '任务不存在。'})
                return self.send(200, job.snapshot())
            if len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'download':
                job = app.jobs.get(parts[2])
                if not job or job.owner != owner:
                    return self.send(404, {'error': '任务不存在。'})
                if job.snapshot()['status'] == 'running':
                    return self.send(409, {'error': '请等待实验结束或停止后下载。'})
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
                    # Only experiment data/reports: keys never enter this directory.
                    for f in job.directory.rglob('*'):
                        if f.is_file() and f.suffix in ('.json', '.jsonl', '.csv', '.md'):
                            z.write(f, f.relative_to(job.directory))
                return self.send(200, buffer.getvalue(), 'application/zip', attachment='generation-experiment.zip')
            self.send(404, {'error': '未找到。'})

        def do_POST(self):
            owner = self.owner()
            # Same-origin JSON requests; cross-origin forms cannot launch charged runs.
            origin = self.headers.get('Origin')
            if not owner or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send(403, {'error': '请从当前页面提交。'})
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                return self.send(403, {'error': '来源不匹配。'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 10_000_000:
                    return self.send(413, {'error': '输入大小须小于 10 MB。'})
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError('请求格式无效。')
                if self.path == '/api/jobs':
                    return self.send(202, app.start(owner, data).snapshot())
                parts = self.path.strip('/').split('/')
                if len(parts) == 4 and parts[:2] == ['api', 'jobs'] and parts[3] == 'stop':
                    job = app.jobs.get(parts[2])
                    if not job or job.owner != owner:
                        return self.send(404, {'error': '任务不存在。'})
                    job.stop.set()
                    return self.send(200, {'message': '停止已请求；正在执行的 API 请求结束后停止。'})
                self.send(404, {'error': '未找到。'})
            except (ValueError, KeyError, TypeError):
                # Validation errors are intentionally generic: never echo user inputs/keys.
                self.send(400, {'error': '配置或数据格式无效：检查 HTTPS API 地址、模型、样本数、方法和人工参考；参考指标还需安装依赖。'})
            except Exception:
                self.send(500, {'error': '无法启动任务，请检查服务器配置。'})
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8080)
    args = p.parse_args()
    httpd = server(args.host, args.port)
    print(f'Experiment UI listening on http://{args.host}:{httpd.server_port}', flush=True)
    httpd.serve_forever()
