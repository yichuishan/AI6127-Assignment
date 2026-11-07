// node tests/experiment_web_ui.cjs -- frontend rendering without dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const root = path.resolve(__dirname,'..');
const html = fs.readFileSync(path.join(root,'web/index.html'),'utf8');
const code = fs.readFileSync(path.join(root,'web/app.js'),'utf8');
class Element {
 constructor(){this.value='';this.textContent='';this.children=[];this.handlers={};this.files=[];this.classList={toggle(){}};}
 replaceChildren(...values){this.children=values;}
 append(...values){this.children.push(...values);}
 addEventListener(name,fn){this.handlers[name]=fn;}
 setAttribute(name,value){this[name]=value;}
}
const elements = new Map([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>[m[1],new Element()]));
const get = id => {assert.ok(elements.has(id),'Missing HTML ID: '+id);return elements.get(id);};
for(const [id,value] of Object.entries({limit:'3',repeats:'1','max-claims':'4','base-url':'https://api.deepseek.com',
 model:'deepseek-v4-pro',thinking:'auto','max-output-tokens':'8192',timeout:'180'})) get(id).value=value;
const arms=['baseline','improved','verify_delete','verify_revise'].map(value=>({value}));
const requests=[];
const job={id:'fixture',status:'completed',completed:1,total:1,failed:0,scored:1,api_calls:1,stage:'ROUGE 评分完成',
 started_at:100,finished_at:101,events:[{case_id:'hybrid_rerank/q01',approach:'baseline',status:'completed',
 metrics:{rouge1_f1:.5,rouge2_f1:.25,rougeL_f1:.4},summary:'Generated answer.',reference:'Reviewed reference answer.'}],
 aggregates:[{approach:'baseline',rouge1_f1:.5,rouge2_f1:.25,rougeL_f1:.4}]};
const ctx=vm.createContext({document:{getElementById:get,querySelectorAll:()=>arms,createElement:()=>new Element(),createTextNode:text=>({textContent:text})},
 fetch:async(url,config)=>{requests.push({url,config});return {ok:true,json:async()=>url==='/api/info'?{cases:100,reference_count:100,evaluation_method:'rouge_1_2_l_human_references_v1',dataset_status:'Fixed references',jobs:[]}:job};},
 setTimeout:()=>1,clearTimeout(){},Date,console});
const run=s=>vm.runInContext(s,ctx);
const text=e=>e.textContent+' '+e.children.map(text).join(' ');
async function main(){
 run(code);await new Promise(resolve=>setImmediate(resolve));
 assert.match(get('dataset-info').textContent,/100 条人工参考答案/);
 assert.match(get('call-estimate').textContent,/评分 0 次模型调用/);
 ctx.fixture=job;run('render(fixture)');
 assert.equal(get('aggregates').children[0].children.length,4);
 assert.equal(get('aggregates').children[0].children[1].textContent,'50.0%');
 assert.equal(get('aggregates').children[0].children[3].textContent,'40.0%');
 assert.match(text(get('events')),/ROUGE-L F1 40.0%/);
 assert.match(text(get('events')),/Reviewed reference answer/);
 assert.doesNotMatch(text(get('events')),/事实 ID/);
 assert.equal(get('download').href,'/api/jobs/fixture/download');
 ctx.fixture={...job,id:'new-job',status:'running',events:[],aggregates:[]};run('render(fixture)');
 assert.doesNotMatch(text(get('events')),/Generated answer/);
 get('api-key').value='temporary-fixture-key';
 await get('experiment-form').handlers.submit({preventDefault(){}});
 const submitted=JSON.parse(requests.find(r=>r.url==='/api/jobs').config.body);
 assert.equal(submitted.api_key,'temporary-fixture-key');
 assert.equal(submitted.evaluation_method,'rouge_1_2_l_human_references_v1');
 assert.ok(!Object.keys(submitted).some(k=>/reference|judge|fact_candidates/.test(k)));
 assert.equal(get('api-key').value,'');
 assert.match(html,/ROUGE-1 F1/);
 assert.doesNotMatch(html,/事实 ID 匹配评测/);
 console.log('ROUGE frontend: reference counts, macro F1 rendering, reference display, clean new-run state and submission passed.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
