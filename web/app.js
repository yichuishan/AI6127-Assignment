const $ = id => document.getElementById(id);
const names={baseline:'A · Baseline',improved:'B · Evidence quotes',verify_delete:'C · Verify & delete',verify_revise:'D · Verify & revise'};
let activeJob=null,pollTimer=null;
const rate=x=>typeof x==='number'&&Number.isFinite(x)?`${(x*100).toFixed(1)}%`:'—';
function alertMessage(message){$('alert').textContent=message;$('alert').hidden=!message;}
function number(id){return $(id).value.trim()===''?null:Number($(id).value);}
function estimate(){
 const arms=[...document.querySelectorAll('[name=arm]:checked')].map(x=>x.value);
 const n=number('limit')*number('repeats'),claims=number('max-claims');
 const calls=n*arms.reduce((a,x)=>a+(x==='verify_delete'?1+claims:x==='verify_revise'?1+3*claims:1),0);
 $('estimate').textContent=`${n*arms.length} 项实验输出`;
 $('call-estimate').textContent=`生成请求上限 ${calls} 次 · 评分 0 次模型调用`;
}
async function request(path,body){
 const response=await fetch(path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const data=await response.json();if(!response.ok)throw new Error(data.error||'请求失败');return data;
}
function render(job){
 activeJob=job;const running=job.status==='running',pct=job.total?Math.floor(job.completed/job.total*100):0;
 $('percentage').replaceChildren(document.createTextNode(String(pct)),Object.assign(document.createElement('span'),{textContent:'%'}));
 $('progress').value=pct;$('completed').textContent=`${job.completed} / ${job.total||'—'}`;
 $('api-calls').textContent=job.api_calls;$('failed').textContent=job.failed;$('scored').textContent=job.scored||0;
 $('stage').textContent=job.stop_reason||job.stage;
 $('status').textContent=({running:'运行中',completed:job.failed?'已结束 · 有失败项':'已完成',stopped:'已停止',failed:'运行失败'})[job.status]||job.status;
 const elapsed=Math.max(0,Math.floor((job.finished_at||Date.now()/1000)-job.started_at));
 $('elapsed').textContent=`${String(Math.floor(elapsed/60)).padStart(2,'0')}:${String(elapsed%60).padStart(2,'0')}`;
 $('config-fields').disabled=running;$('start').disabled=running;$('stop').disabled=!running;
 $('start').textContent=running?'实验运行中…':'开始新实验 ↗';
 $('download').classList.toggle('disabled',running);$('download').setAttribute('aria-disabled',String(running));
 if(!running)$('download').href=`/api/jobs/${job.id}/download`;
 if(job.events.length){
  $('events').replaceChildren();
  [...job.events].reverse().forEach(event=>{
   const article=document.createElement('article');article.className='event'+(event.status==='failed'?' error':'');
   const head=document.createElement('div');head.className='event-head';
   const b=document.createElement('b');b.textContent=names[event.approach];
   const id=document.createElement('small');id.textContent=event.case_id;
   const status=document.createElement('span');status.textContent=event.status==='failed'?`失败 · ${event.error_type}`:'✓ 算法评分完成';
   head.append(b,id,status);article.append(head);
   const add=text=>{const p=document.createElement('p');p.textContent=text;article.append(p);};
   if(event.error_message)add(`${event.error_stage||'generation'} · ${event.error_code||''}：${event.error_message}`);
   if(event.status==='completed'){
    const m=event.metrics;add(`P ${rate(m.fact_selection_precision)} · R ${rate(m.fact_selection_recall)} · F1 ${rate(m.fact_selection_f1)} · 单元覆盖 ${rate(m.coverage_unit_recall)} · 正负覆盖 ${rate(m.polarity_contrast_coverage)}`);
    add(`事实 ID：${event.selected_fact_ids.join(', ')||'空选择'}`);
   }
   if(event.summary)add(event.summary);
   $('events').append(article);
  });
 }
 $('aggregate-panel').hidden=!job.aggregates?.length;
 if(job.aggregates?.length){
  $('aggregates').replaceChildren();
  job.aggregates.forEach(row=>{
   const tr=document.createElement('tr');
   [names[row.approach].split(' · ')[0],...['fact_selection_precision','fact_selection_recall','fact_selection_f1','coverage_unit_recall','polarity_contrast_coverage'].map(k=>rate(row[k]))].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});
   $('aggregates').append(tr);
  });
 }
 if(job.status==='failed')alertMessage(job.error_message||'无法完成实验，请检查配置。');
 clearTimeout(pollTimer);if(running)pollTimer=setTimeout(poll,1500);
}
async function poll(){if(!activeJob)return;try{render(await request(`/api/jobs/${activeJob.id}`));}catch(e){alertMessage('进度连接暂时中断，正在重连。');pollTimer=setTimeout(poll,4000);}}
$('experiment-form').addEventListener('input',estimate);
$('smoke-preset').onclick=()=>{$('limit').value=3;estimate();};
$('full-preset').onclick=()=>{$('limit').value=100;estimate();};
$('deepseek-preset').onclick=()=>{$('base-url').value='https://api.deepseek.com';$('model').value='deepseek-v4-pro';$('thinking').value='auto';$('max-output-tokens').value=8192;$('timeout').value=180;estimate();};
$('experiment-form').addEventListener('submit',async event=>{
 event.preventDefault();alertMessage('');$('start').disabled=true;
 try{
  const arms=[...document.querySelectorAll('[name=arm]:checked')].map(x=>x.value);if(!arms.length)throw new Error('请至少选择一种方法。');
  const file=$('dataset-file').files[0];if(file&&file.size>9000000)throw new Error('文件须小于9 MB。');
  const body={base_url:$('base-url').value,model:$('model').value,api_key:$('api-key').value,
   limit:number('limit'),repeats:number('repeats'),max_claims:number('max-claims'),approaches:arms,
   max_output_tokens:number('max-output-tokens'),timeout:number('timeout'),thinking:$('thinking').value,
   input_price:number('input-price'),output_price:number('output-price'),dataset_jsonl:file?await file.text():''};
  const job=await request('/api/jobs',body);$('api-key').value='';render(job);
 }catch(e){alertMessage(e.message);$('start').disabled=false;}
});
$('stop').onclick=async()=>{if(!activeJob)return;$('stop').disabled=true;try{const result=await request(`/api/jobs/${activeJob.id}/stop`,{});$('stage').textContent=result.message;}catch(e){alertMessage(e.message);$('stop').disabled=false;}};
(async()=>{estimate();try{const info=await request('/api/info');$('dataset-info').textContent=`${info.cases} 个问题 · ${info.fact_count} 个标注事实 · 程序匹配评分`;$('source-status').textContent=info.dataset_status;const jobs=info.jobs.sort((a,b)=>b.started_at-a.started_at);if(jobs.length)render(jobs[0]);}catch(e){alertMessage('无法读取实验服务，请刷新或重新启动服务。');}})();
