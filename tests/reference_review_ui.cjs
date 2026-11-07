// Optional browser-state tests: node tests/reference_review_ui.cjs (no packages).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const page = fs.readFileSync(path.join(root, 'data/reference_summaries/drafts/review.html'), 'utf8');
const dataText = page.match(/<script id="draft-data"[^>]*>([\s\S]*?)<\/script>/)[1];
const program = page.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'data/reference_summaries/drafts/manifest.json')));

function browser(saved = {}) {
  class Element {
    constructor() { this.children = []; this.value = ''; this.textContent = ''; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    click() { if (this.onclick) this.onclick(); }
  }
  const elements = new Map();
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  get('draft-data').textContent = dataText;
  get('filter').value = 'all';
  const storage = new Map(Object.entries(saved));
  const context = vm.createContext({
    document: {getElementById: get, createElement: () => new Element()},
    localStorage: {getItem: k => storage.get(k) ?? null, setItem: (k,v) => storage.set(k,v)},
    URL, Blob, setTimeout: () => {}, console,
  });
  vm.runInContext(program, context);
  const run = code => vm.runInContext(code, context);
  return {get, run, storage, context};
}
const clone = value => JSON.parse(JSON.stringify(value));
let checks = 0;
function check(name, test) { test(); checks++; console.log('ok - ' + name); }

async function main() {
  let b = browser();
  const rows = clone(b.run('DATA'));
  const r = rows.find(r => r.query_id === 'q63');
  const index = rows.indexOf(r);
  const revision = b.run('REVISION');
  const legacyRevision = b.run('LEGACY_REVISIONS[0]');
  const legacyKey = 'ai6127_reference_review_' + legacyRevision;
  const key = b.run('KEY');

  check('100 pending rows with vote-aware draft and source headings', () => {
    assert.equal(rows.length, 100);
    b.run(`show(${index})`);
    assert.match(b.get('reference-zh').value, /高赞评论/);
    assert.match(b.get('reference').value, /highly upvoted review/);
    const heading = b.get('claim-evidence').children[0].children.find(e => e.children?.[0]?.textContent?.includes('helpful votes'));
    assert.match(heading.children[0].textContent, /224 次有用投票/);
    assert.match(heading.children[1].textContent, /2020-04-08/);
    assert.equal(b.run('DATA.filter(r=>ready(r,state(r))).length'), 0);
  });
  check('versioned export includes matching metadata and English draft revisions', () => {
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(record.dataset_revision, revision);
    assert.equal(record.review_ui_version, 'bilingual_synthesis_v4');
    assert.equal(record.english_draft_revision, manifest.english_draft_revision.slice(0,20));
    assert.equal(record.review_metadata_sha256, r.review_metadata_sha256);
    assert.equal(record.reference_policy, 'coherent_synthesis_v1');
  });
  check('approving unchanged bilingual draft yields an English-ready reference', () => {
    b.get('reviewer').value = 'Tester'; b.get('decision').value = 'approved';
    b.run('save()');
    assert.equal(b.run(`record(DATA[${index}]).human_reviewed`), true);
    assert.equal(b.run(`record(DATA[${index}]).reference`), r.draft_reference_summary);
  });
  check('approved Chinese edits wait for translation and cannot enter English export', () => {
    b.get('reference-zh').value += ' 补充核验修改。';
    b.get('edit-language').value = 'zh'; b.get('decision').value = 'edited'; b.run('save()');
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(record.translation_status, 'pending_translation');
    assert.equal(record.content_reviewed, true);
    assert.equal(record.human_reviewed, false);
    assert.equal(record.reference, null);
  });

  const old = {review_status:'approved', english_text:r.legacy_drafts[1].en,
    chinese_text:r.legacy_drafts[1].zh, reviewer:'Original reviewer', review_notes:'Keep this note'};
  check('saved previous-version approvals migrate to revised text and pending status', () => {
    b = browser({[legacyKey]:JSON.stringify({[r.id]:old})});
    b.run(`show(${index})`);
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(b.get('reference').value, r.draft_reference_summary);
    assert.equal(b.get('reference-zh').value, r.draft_reference_summary_zh);
    assert.equal(record.review_status, 'pending'); assert.equal(record.human_reviewed, false);
    assert.equal(record.reviewer, 'Original reviewer');
    assert.match(record.review_notes, /Keep this note/);
    assert.equal(JSON.parse(b.storage.get(legacyKey))[r.id].review_status, 'approved');
  });
  check('migration preserves personal edits and clears obsolete translation approval', () => {
    const edited = {...old, english_text:'My own English edit.', chinese_text:'我的中文修改。',
      translated_by:'Translator', translated_source_zh:'我的中文修改。', translated_reference_en:'My own English edit.'};
    b = browser({[legacyKey]:JSON.stringify({[r.id]:edited})}); b.run(`show(${index})`);
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(record.edited_draft_en, 'My own English edit.');
    assert.equal(record.edited_draft_zh, '我的中文修改。');
    assert.equal(record.translated_by, ''); assert.equal(record.review_status, 'pending');
  });
  check('older bilingual-v2 browser progress also restores into the new synthesis draft', () => {
    const v2Revision = b.run("LEGACY_REVISIONS.find(v=>v.endsWith('_bilingual_v2'))");
    const prior = {...old, english_text:r.legacy_drafts[0].en, chinese_text:r.legacy_drafts[0].zh};
    b = browser({['ai6127_reference_review_' + v2Revision]:JSON.stringify({[r.id]:prior})});
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(record.edited_draft_en, r.draft_reference_summary);
    assert.equal(record.edited_draft_zh, r.draft_reference_summary_zh);
    assert.equal(record.review_status, 'pending');
  });
  check('current-version browser approvals survive reload', () => {
    const approved = {...old, english_text:r.draft_reference_summary, chinese_text:r.draft_reference_summary_zh};
    b = browser({[key]:JSON.stringify({[r.id]:approved})});
    assert.equal(b.run(`record(DATA[${index}]).human_reviewed`), true);
  });

  async function importRows(records) {
    b.get('import').files = [{text:async()=>records.map(r=>JSON.stringify(r)).join('\n')}];
    await b.get('import').onchange();
  }
  b = browser();
  await importRows([{id:r.id, context_sha256:r.context_sha256, dataset_revision:legacyRevision,
    review_status:'approved', reviewer:'Legacy', edited_draft_en:r.legacy_drafts[1].en,
    edited_draft_zh:r.legacy_drafts[1].zh}]);
  check('imported legacy approvals also require fresh confirmation', () => {
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(record.review_status, 'pending'); assert.equal(record.human_reviewed, false);
    assert.equal(record.edited_draft_en, r.draft_reference_summary);
  });
  await importRows([{id:r.id, context_sha256:r.context_sha256, dataset_revision:b.run('LEGACY_REVISIONS.at(-1)'),
    review_status:'approved', reviewer:'English-only', edited_draft:r.legacy_drafts[0].en}]);
  check('English-only exports restore without losing reviewer details', () => {
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(record.reviewer, 'English-only'); assert.equal(record.review_status, 'pending');
    assert.equal(record.edited_draft_en, r.draft_reference_summary);
  });
  const current = {id:r.id, context_sha256:r.context_sha256, dataset_revision:revision,
    review_metadata_sha256:r.review_metadata_sha256, review_status:'approved', reviewer:'Current',
    edited_draft_en:r.draft_reference_summary, edited_draft_zh:r.draft_reference_summary_zh};
  await importRows([current]);
  check('valid current exports retain approval', () => {
    assert.equal(b.run(`record(DATA[${index}]).human_reviewed`), true);
  });
  await importRows([{...current, review_metadata_sha256:'changed', reviewer:'Must not overwrite'}]);
  check('mismatched helpful-vote metadata is rejected without overwriting existing records', () => {
    assert.match(b.get('message').textContent, /投票元数据版本不匹配/);
    assert.equal(b.run(`record(DATA[${index}]).reviewer`), 'Current');
  });
  const translated = {...current, review_status:'edited', review_language:'zh',
    edited_draft_en:'A reviewed English translation.', edited_draft_zh:'经过核验的中文。',
    translated_by:'Translator', translated_source_zh:'经过核验的中文。',
    translated_reference_en:'A reviewed English translation.'};
  await importRows([translated]);
  check('current-version aligned translations remain English-ready', () => {
    const record = clone(b.run(`record(DATA[${index}])`));
    assert.equal(record.human_reviewed, true); assert.equal(record.translation_status, 'translated');
    assert.equal(record.reference, 'A reviewed English translation.');
  });
  console.log(`${checks} UI workflow checks passed.`);
}
main().catch(e => {console.error(e); process.exitCode=1;});
