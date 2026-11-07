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

from rag_generator import OpenAICompatibleClient, safe_error
from generation_experiment_design import (APPROACHES, EXPERIMENT_PROTOCOL, load_split,
    select_cases, validate_cases)
from context_compression import DEFAULT_RATIO
from run_generation_experiments import jsonl, run, benchmark, aggregate
from evaluate_generation import EVALUATION_METHOD, load_references
from experiment_execution import EXECUTION_PROTOCOL, DEFAULT_CONCURRENCY, DEFAULT_RETRIES, MAX_CONCURRENCY
from generation_paths import BENCHMARK, RESULTS_DIR

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = BENCHMARK


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
    def __init__(self, job, key, base_url, factory, *, max_tokens=8192, thinking='auto', timeout=180):
        self.job = job
        self.transport = factory(base_url=base_url, api_key=key, timeout=timeout,max_tokens=max_tokens,thinking=thinking)

    def complete(self, messages, model, temperature):
        job = self.job
        with job.lock:
            job.state.update(stage='并发生成总结', api_calls=job.state['api_calls'] + 1,
                             active_requests=job.state['active_requests']+1)
            job.state['peak_active_requests']=max(job.state['peak_active_requests'],job.state['active_requests'])
        try:
            response = self.transport.complete(messages, model, temperature)
            with job.lock:job.state['successful_api_calls'] += 1
            return response
        finally:
            with job.lock:job.state['active_requests'] -= 1


class Job:
    def __init__(self, owner, data, directory, factory):
        self.owner, self.directory = owner, directory
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.state = {'id': directory.name, 'status': 'running', 'completed': 0, 'total': 0,
                      'experiment_protocol':EXPERIMENT_PROTOCOL,'split':data['split'],
                      'execution_protocol':EXECUTION_PROTOCOL,'concurrency':data['concurrency'],
                      'max_retries':data['max_retries'],'retry_requests':0,
                      'active_requests':0,'peak_active_requests':0,
                      'compression_ratio':data['compression_ratio'],'model':data['model'],
                      'failed': 0, 'scored':0, 'aggregates':[], 'api_calls': 0, 'successful_api_calls': 0,
                      'stage': '准备输入', 'events': [], 'started_at': time.time()}
        self.thread = threading.Thread(target=self.execute, args=(data, factory), daemon=True)

    def execute(self, data, factory):
        try:
            key = data.pop('api_key')
            client = LiveClient(self,key,data['base_url'],factory,max_tokens=data['max_output_tokens'],
                                thinking=data['thinking'],timeout=data['timeout'])
            key = None
            cases = data.pop('cases')
            (self.directory/'input.jsonl').write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
            with self.lock:
                self.state['total'] = len(cases)*data['repeats']*len(data['approaches'])
            finished_rows=[]
            def activity(event):
                with self.lock:
                    if event['type']=='retry_started':self.state['retry_requests'] += 1
                    elif event['type']=='retry_wait':
                        self.state['stage']=f"{event['case_id']} · 准备第 {event['next_attempt']} 次请求"
                    elif event['type']=='fatal':
                        self.state['stop_reason']='配置或账户错误，已停止新请求：'+event['error_message']
                        self.stop.set()
            def progress(row, completed, total):
                finished_rows.append(row)
                with self.lock:
                    self.state.update(completed=completed,total=total,stage='任务完成，继续处理并发请求',
                                      aggregates=aggregate(finished_rows))
                    self.state['failed'] += row['status']=='failed'
                    self.state['scored'] += row['status']=='completed'
                    self.state['events'].append({'case_id':row['case_id'],'approach':row['approach'],
                        'status':row['status'],'error_type':row.get('error_type'),
                        'error_message':row.get('error_message'),'error_code':row.get('error_code'),
                        'http_status':row.get('http_status'),'error_stage':row.get('error_stage'),
                        'metrics':row['metrics'],
                        'attempt_count':row['attempt_count'],'retry_count':row['retry_count'],
                        'reference':row.get('reference',''),
                        'summary':row.get('result',{}).get('summary','')[:1800]})
                    self.state['events']=self.state['events'][-30:]
            public_config={k:data[k] for k in ('base_url','max_output_tokens','thinking','timeout','input_price','output_price')}
            rows=run(cases,self.directory/'results',client=client,model=data['model'],
                approaches=data['approaches'],repeats=data['repeats'],max_claims=data['max_claims'],
                input_price=data['input_price'],output_price=data['output_price'],config=public_config,
                progress=progress,stop_requested=self.stop.is_set,
                split=data['split'],compression_ratio=data['compression_ratio'],
                concurrency=data['concurrency'],max_retries=data['max_retries'],activity=activity)
            with self.lock:
                self.state.update(status='stopped' if self.stop.is_set() else 'completed',
                    stage='已停止' if self.stop.is_set() else 'ROUGE 总结实验结束',aggregates=aggregate(rows))
        except Exception as error:
            with self.lock:
                self.state.update(status='failed',stage='运行失败',**safe_error(error))
        finally:
            data.clear()
            with self.lock:self.state['finished_at']=time.time()

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))


class Application:
    def __init__(self, run_root=None, factory=OpenAICompatibleClient):
        self.run_root = Path(run_root or RESULTS_DIR)
        self.run_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.factory, self.jobs, self.sessions, self.lock = factory, {}, set(), threading.Lock()

    def start(self, owner, data):
        allowed={'api_key','base_url','model','repeats','max_output_tokens','timeout','thinking',
                 'max_claims','limit','approaches','input_price','output_price','dataset_jsonl','evaluation_method',
                 'experiment_protocol','split','compression_ratio','concurrency','max_retries','execution_protocol'}
        if set(data)-allowed:raise ValueError('Unsupported experiment option')
        if data.get('evaluation_method',EVALUATION_METHOD)!=EVALUATION_METHOD:
            raise ValueError('评测版本不匹配，请重启服务并刷新网页。')
        if data.get('experiment_protocol')!=EXPERIMENT_PROTOCOL:
            raise ValueError('实验已改为少样本提示 × 上下文压缩，请重启服务并刷新网页。')
        if data.get('execution_protocol')!=EXECUTION_PROTOCOL:
            raise ValueError('并发与重试版本不匹配，请重启服务并刷新网页。')
        key=data.get('api_key','')
        if not isinstance(key,str) or not key.strip() or len(key)>4096:raise ValueError('请输入 API Key。')
        data['api_key']=key.strip();data['base_url']=endpoint(data.get('base_url',''))
        model=data.get('model','')
        if not isinstance(model,str) or not model.strip() or len(model)>200:raise ValueError('模型名称无效。')
        data['model']=model.strip()
        data['repeats']=integer(data.get('repeats',1),1,5)
        data['concurrency']=integer(data.get('concurrency',DEFAULT_CONCURRENCY),1,MAX_CONCURRENCY)
        data['max_retries']=integer(data.get('max_retries',DEFAULT_RETRIES),0,2)
        data['max_output_tokens']=integer(data.get('max_output_tokens',8192),256,131072)
        data['timeout']=integer(data.get('timeout',180),10,600)
        data['thinking']=data.get('thinking','auto')
        if data['thinking'] not in ('auto','disabled','enabled'):raise ValueError('思考模式无效。')
        data['max_claims']=integer(data.get('max_claims',4),1,12)
        data['split']=data.get('split','test')
        if data['split'] not in ('test','development'):raise ValueError('数据划分无效。')
        available=select_cases(jsonl(DEFAULT_DATASET),data['split'])
        limit=integer(data.get('limit',3),1,len(available))
        ratio=data.get('compression_ratio',DEFAULT_RATIO)
        if type(ratio) not in (float,int) or not math.isfinite(ratio) or not .2<=ratio<=1:
            raise ValueError('评论文本保留比例应在 0.2–1 之间。')
        data['compression_ratio']=ratio
        arms=data.get('approaches',list(APPROACHES))
        if not isinstance(arms,list) or not arms or len(set(arms))!=len(arms) or any(a not in APPROACHES for a in arms):
            raise ValueError('实验方法无效。')
        data['approaches']=arms
        for name in ('input_price','output_price'):
            value=data.get(name)
            if value is not None and (type(value) not in (float,int) or not math.isfinite(value) or value<0):raise ValueError('价格无效。')
            data[name]=value
        text=data.pop('dataset_jsonl','')
        cases=[json.loads(line) for line in text.splitlines() if line.strip()] if text else available
        cases=cases[:limit]
        if not cases or any(not isinstance(c,dict) or not isinstance(c.get('id'),str) for c in cases) or len({c['id'] for c in cases})!=len(cases):
            raise ValueError('案例 ID 无效或重复。')
        benchmark(cases)
        validate_cases(cases,data['split'])
        # Discard any extra uploaded fields; persist the known canonical cases only.
        canonical={c['id']:c for c in jsonl(DEFAULT_DATASET)}
        data['cases']=[canonical[c['id']] for c in cases]
        with self.lock:
            if any(j.snapshot()['status']=='running' for j in self.jobs.values()):raise ValueError('已有实验运行中。')
            directory=self.run_root/secrets.token_hex(16);directory.mkdir(mode=0o700)
            job=Job(owner,data,directory,self.factory);self.jobs[directory.name]=job;job.thread.start()
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
                refs=load_references()
                split=load_split()
                return self.send(200, {'cases': len(cases), 'reference_count':len(refs),
                    'experiment_protocol':EXPERIMENT_PROTOCOL,
                    'execution_protocol':EXECUTION_PROTOCOL,'max_concurrency':MAX_CONCURRENCY,
                    'splits':{'test':len(split['test_ids']),'development':len(split['development_ids'])-len(split['example_ids'])},
                    'development_total':len(split['development_ids']),'example_ids':split['example_ids'],
                    'evaluation_method':EVALUATION_METHOD,'reference_version':next(iter(refs.values()))['dataset_revision'],
                    'dataset_status': '固定划分：20 题开发（2 题作示例、18 题可调参），80 题测试。少样本仅使用开发示例，当前题参考答案不发送给模型。压缩从原始固定评论中抽取整句并保留来源。',
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
                    return self.send(200, {'message': '停止已请求；不再发起新请求，当前并发请求结束后保存结果。'})
                self.send(404, {'error': '未找到。'})
            except (ValueError, KeyError, TypeError):
                # Validation errors are intentionally generic: never echo user inputs/keys.
                self.send(400, {'error': '配置或数据格式无效：请刷新新版页面，仅填写生成模型；上传案例须与已标注的100题基准一致。'})
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
