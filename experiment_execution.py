"""Bounded retry policy and per-attempt accounting for experiment requests."""
import math
import time

from rag_generator import GenerationError, safe_error

EXECUTION_PROTOCOL = 'concurrent_retry_v1'
DEFAULT_CONCURRENCY = 20
MAX_CONCURRENCY = 500
DEFAULT_RETRIES = 2
FATAL_CODES = {'missing_key','http_400','http_401','http_402','http_403','http_404','http_422'}
RETRYABLE_CODES = {'timeout','network_error','http_429','http_5xx',
    'response_schema','output_truncated','empty_content','invalid_json','invalid_claims','invalid_citations'}


class ExecutionCancelled(Exception):
    pass


class AttemptClient:
    def __init__(self,client,attempt,stop_requested,activity):
        self.client,self.attempt,self.stop_requested,self.activity=client,attempt,stop_requested,activity
        self.calls=[]

    def complete(self,messages,model,temperature):
        if self.stop_requested():
            raise ExecutionCancelled()
        if self.attempt>1 and self.activity:
            self.activity({'type':'retry_started','attempt':self.attempt})
        started=time.perf_counter()
        call={'usage':{},'usage_complete':False}
        self.calls.append(call)
        try:
            response=self.client.complete(messages,model,temperature)
            usage=response.get('usage') or {}
            if not isinstance(usage,dict):usage={}
            call['usage']={k:usage[k] for k in ('prompt_tokens','completion_tokens')
                           if type(usage.get(k)) is int and usage[k]>=0}
            call['usage_complete']=len(call['usage'])==2
            return response
        finally:
            call['latency_seconds']=round(time.perf_counter()-started,6)


def request_totals(attempts,input_price=None,output_price=None):
    calls=[call for attempt in attempts for call in attempt['calls']]
    complete=bool(calls) and all(c['usage_complete'] for c in calls)
    totals={'model_calls':len(calls),'usage_complete':complete}
    for k in ('prompt_tokens','completion_tokens'):
        known=[c['usage'][k] for c in calls if k in c['usage']]
        totals[k]=sum(known) if complete else None
        totals['observed_'+k]=sum(known) if known else None
    totals['estimated_cost_usd']=(totals['prompt_tokens']*input_price+totals['completion_tokens']*output_price)/1e6 if (
        complete and input_price is not None and output_price is not None) else None
    return totals


def wait_for_retry(delay,stop_requested):
    deadline=time.monotonic()+delay
    while not stop_requested():
        left=deadline-time.monotonic()
        if left<=0:return True
        time.sleep(min(.1,left))
    return False


def execute_with_retries(operation,client,*,max_retries=DEFAULT_RETRIES,retry_backoff=1,
                         stop_requested=None,activity=None,input_price=None,output_price=None):
    """Retry the same prompt on transient/format errors; never select by ROUGE."""
    if type(max_retries) is not int or not 0<=max_retries<=2:
        raise ValueError('Maximum retries must be between 0 and 2')
    if type(retry_backoff) not in (float,int) or not math.isfinite(retry_backoff) or not 0<=retry_backoff<=60:
        raise ValueError('Invalid retry backoff')
    stopped=stop_requested or (lambda:False)
    attempts=[];started=time.perf_counter();last_error={};result=None;interrupted=False
    for number in range(1,max_retries+2):
        if stopped():
            interrupted=True;break
        transport=AttemptClient(client,number,stopped,activity)
        attempt={'attempt':number,'status':'completed','calls':transport.calls}
        tick=time.perf_counter()
        try:
            result=operation(transport)
        except ExecutionCancelled:
            interrupted=True;break
        except Exception as error:
            last_error=safe_error(error)
            attempt.update(status='failed',**last_error)
            attempt['retryable']=isinstance(error,GenerationError) and error.code in RETRYABLE_CODES
            if attempt['retryable'] and number<=max_retries:
                delay=max(retry_backoff*2**(number-1),getattr(error,'retry_after_seconds',None) or 0)
                attempt['retry_delay_seconds']=delay
        finally:
            attempt['latency_seconds']=round(time.perf_counter()-tick,6)
        attempts.append(attempt)
        if result is not None:
            break
        if not attempt.get('retryable') or number>max_retries:
            break
        if activity:
            activity({'type':'retry_wait','attempt':number,'next_attempt':number+1,
                'delay_seconds':attempt['retry_delay_seconds'],**last_error})
        if not wait_for_retry(attempt['retry_delay_seconds'],stopped):
            interrupted=True;break
    if not attempts:
        return None
    info={'status':'completed' if result is not None else 'failed','attempts':attempts,
          'attempt_count':len(attempts),'retry_count':max(0,len(attempts)-1),
          'request_totals':request_totals(attempts,input_price,output_price),
          'task_latency_seconds':round(time.perf_counter()-started,6)}
    if result is not None:info['result']=result
    else:info.update(last_error)
    if interrupted:info['retry_interrupted']=True
    return info
