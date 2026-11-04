const $ = id => document.getElementById(id);
let activeJob = null, pollTimer = null;
const names = {baseline:'A · Baseline',improved:'B · Evidence quotes',verify_delete:'C · Verify & delete',verify_revise:'D · Verify & revise'};
function alertMessage(message) { $('alert').textContent=message; $('alert').hidden=!message; }
function number(id) { const v=$(id).value.trim(); return v===''?null:Number(v); }
function estimate() {
  const arms=[...document.querySelectorAll('[name=arm]:checked')].map(x=>x.value);
  const n=Number($('limit').value)*Number($('repeats').value), claims=Number($('max-claims').value);
  const calls=n*arms.reduce((a,x)=>a+(x==='verify_delete'?1+claims:x==='verify_revise'?1+3*claims:1),0);
  $('estimate').textContent=`${n*arms.length} 项实验输出`;
  $('call-estimate').textContent=`生成请求上限 ${calls} 次${$('judge-model').value.trim()?`，另有 ${n*arms.length} 次最终评估`:''}`;
}
async function request(path,body) {
  const response=await fetch(path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok) throw new Error(data.error||'请求失败，请重试。');
  return data;
}
function render(job) {
  activeJob=job;
  const running=job.status==='running', pct=job.total?Math.floor(job.completed/job.total*100):0;
  $('percentage').replaceChildren(document.createTextNode(String(pct)),Object.assign(document.createElement('span'),{textContent:'%'}));
  $('progress').value=pct;
  $('completed').textContent=`${job.completed} / ${job.total||'—'}`;
  $('api-calls').textContent=job.api_calls;
  $('failed').textContent=job.failed;
  $('judge-failed').textContent=job.judge_failed||0;
  $('stage').textContent=job.stop_reason||job.stage;
  $('status').textContent=({running:'运行中',completed:job.failed?'已结束 · 有失败项':job.judge_failed?'已结束 · 评测未全部完成':'已完成',stopped:'已停止',failed:'运行失败'})[job.status]||job.status;
  const elapsed=Math.max(0,Math.floor((job.finished_at||Date.now()/1000)-job.started_at));
  $('elapsed').textContent=`${String(Math.floor(elapsed/60)).padStart(2,'0')}:${String(elapsed%60).padStart(2,'0')}`;
  $('config-fields').disabled=running; $('start').disabled=running; $('stop').disabled=!running;
  $('start').textContent=running?'实验运行中…':'开始新实验 ↗';
  $('download').classList.toggle('disabled',running);
  $('download').setAttribute('aria-disabled',String(running));
  if(!running) $('download').href=`/api/jobs/${job.id}/download`;
  if(job.events.length) {
    $('events').replaceChildren();
    [...job.events].reverse().forEach(event=>{
      const article=document.createElement('article'); article.className='event'+(['failed','judge_failed'].includes(event.status)?' error':'');
      const head=document.createElement('div'); head.className='event-head';
      const b=document.createElement('b'); b.textContent=names[event.approach];
      const id=document.createElement('small'); id.textContent=event.case_id;
      const status=document.createElement('span'); status.textContent=event.status==='judge_failed'?'生成成功 · 评测失败':event.status==='failed'?`失败 · ${event.error_type}`:'✓ 已完成';
      head.append(b,id,status); article.append(head);
      if(event.error_message){const p=document.createElement('p');p.textContent=`${event.error_stage?event.error_stage+' · ':''}${event.error_code||''}${event.http_status?' · HTTP '+event.http_status:''}：${event.error_message}`;article.append(p);}
      if(event.summary){const p=document.createElement('p');p.textContent=event.summary;article.append(p);}
      $('events').append(article);
    });
  }
  if(job.status==='failed')alertMessage(job.error_message||`任务未能完成（${job.error_type||'未知错误'}）。请检查配置。`);
  clearTimeout(pollTimer);
  if(running) pollTimer=setTimeout(poll,1500);
}
async function poll() {
  if(!activeJob)return;
  try { render(await request(`/api/jobs/${activeJob.id}`)); }
  catch(error) {alertMessage('进度连接暂时中断，正在重连；服务端任务可能仍在运行。');pollTimer=setTimeout(poll,4000);}
}
$('experiment-form').addEventListener('input',estimate);
$('smoke-preset').onclick=()=>{$('limit').value=3;estimate();};
$('deepseek-preset').onclick=()=>{$('base-url').value='https://api.deepseek.com';$('model').value='deepseek-v4-pro';$('thinking').value='auto';$('max-output-tokens').value=8192;$('timeout').value=180;estimate();};
$('qwen-judge-preset').onclick=()=>{$('judge-model').value='qwen3.8-max';$('judge-thinking').value='disabled';$('judge-max-output-tokens').value=4096;$('judge-timeout').value=300;estimate();};
$('full-preset').onclick=()=>{$('limit').value=100;estimate();};
$('experiment-form').addEventListener('submit',async event=>{
  event.preventDefault();alertMessage('');$('start').disabled=true;
  try {
    const arms=[...document.querySelectorAll('[name=arm]:checked')].map(x=>x.value);
    if(!arms.length)throw new Error('请至少选择一种实验方法。');
    const file=$('dataset-file').files[0];
    if(file&&file.size>9000000)throw new Error('测试文件须小于 9 MB。');
    const body={base_url:$('base-url').value,model:$('model').value,api_key:$('api-key').value,
      judge_model:$('judge-model').value,judge_base_url:$('judge-base-url').value,judge_api_key:$('judge-api-key').value,
      limit:number('limit'),repeats:number('repeats'),max_claims:number('max-claims'),approaches:arms,
      max_output_tokens:number('max-output-tokens'),timeout:number('timeout'),thinking:$('thinking').value,
      judge_max_output_tokens:number('judge-max-output-tokens'),judge_timeout:number('judge-timeout'),judge_thinking:$('judge-thinking').value,
      reference_metrics:['rouge','bertscore'].filter(id=>$(id).checked),dataset_jsonl:file?await file.text():''};
    for(const id of ['input-price','output-price','judge-input-price','judge-output-price'])body[id.replaceAll('-','_')]=number(id);
    const job=await request('/api/jobs',body);
    $('api-key').value='';$('judge-api-key').value='';
    render(job);
  } catch(error){alertMessage(error.message);$('start').disabled=false;}
});
$('stop').onclick=async()=>{
  if(!activeJob)return;
  $('stop').disabled=true;
  try {const data=await request(`/api/jobs/${activeJob.id}/stop`,{});$('stage').textContent=data.message;}
  catch(error){alertMessage(error.message);$('stop').disabled=false;}
};
(async()=>{
  estimate();
  try {
    const info=await request('/api/info');
    $('dataset-info').textContent=`${info.cases} 个问题 · 每题最多 10 条证据 · ${info.references} 条人工参考总结`;
    $('source-status').textContent=info.dataset_status;
    const jobs=info.jobs.sort((a,b)=>b.started_at-a.started_at);
    if(jobs.length)render(jobs[0]);
  } catch(error){alertMessage('无法读取实验服务，请刷新或检查服务是否已启动。');}
})();
