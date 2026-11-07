"""Optional real-browser smoke test. Requires Playwright and a Chromium binary.

Run from the repository: python tests/browser_experiment.py
Scripted model only; uses a temporary localhost port and never a paid API.
"""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_web import Application, server
from evaluate_generation import rescore
from generation_experiment_design import APPROACHES, EXPERIMENT_PROTOCOL


class ScriptedModel:
    def __init__(self,**kwargs):pass

    def complete(self,messages,model,temperature):
        assert 'evidence verifier' not in messages[0]['content']
        payload=json.loads(messages[1]['content'])
        assert 'reference' not in payload
        e=payload['evidence'][0]
        answer={'claims':[{'text':'A review describes the game experience.', 'chunk_ids':[e['chunk_id']]}]}
        return {'content':json.dumps(answer),'usage':{'prompt_tokens':100,'completion_tokens':20}}


def main():
    from playwright.sync_api import sync_playwright, expect
    with tempfile.TemporaryDirectory() as tmp:
        app=Application(Path(tmp)/'runs',factory=ScriptedModel)
        httpd=server('127.0.0.1',0,app)
        worker=threading.Thread(target=httpd.serve_forever,daemon=True);worker.start()
        try:
            with sync_playwright() as p:
                executable=os.getenv('CHROMIUM_PATH') or shutil.which('chromium')
                browser=p.chromium.launch(headless=True,executable_path=executable,
                    args=['--no-sandbox','--disable-dev-shm-usage'])
                page=browser.new_page(viewport={'width':1440,'height':1000},accept_downloads=True)
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(f'http://127.0.0.1:{httpd.server_port}')
                expect(page.locator('#dataset-info')).to_contain_text('80 题测试')
                assert page.locator('[name=arm]').evaluate_all('(xs)=>xs.map(x=>x.value)')==list(APPROACHES)
                assert page.locator('#limit').get_attribute('max')=='80'
                page.locator('#full-preset').click();assert page.locator('#limit').input_value()=='80'
                page.locator('#split').select_option('development')
                page.locator('#full-preset').click();assert page.locator('#limit').input_value()=='18'
                page.locator('#split').select_option('test')
                page.locator('#limit').fill('1')
                page.locator('#api-key').fill('browser-test-secret-not-real')
                page.locator('#start').click()
                expect(page.locator('#status')).to_have_text('已完成',timeout=30000)
                assert page.locator('#api-calls').inner_text()=='4'
                assert page.locator('#aggregates tr').count()==4
                assert page.locator('#events details').count()==4
                assert page.locator('#api-key').input_value()==''
                assert page.evaluate('localStorage.length')==0
                with page.expect_download() as download_info:
                    page.locator('#download').click()
                download=download_info.value
                body=Path(download.path()).read_bytes()
                with zipfile.ZipFile(io.BytesIO(body)) as z:
                    assert 'results/few_shot_examples.json' in z.namelist()
                    manifest=json.loads(z.read('results/manifest.json'))
                    assert manifest['experiment_protocol']==EXPERIMENT_PROTOCOL
                    rows=json.loads(z.read('results/generation_results.json'))
                    assert {r['approach'] for r in rows}==set(APPROACHES)
                    assert rescore(rows)==rows
                    for name in z.namelist():assert b'browser-test-secret-not-real' not in z.read(name)
                page.screenshot(path='/tmp/ai6127-fewshot-web-desktop.png',full_page=True)
                page.set_viewport_size({'width':390,'height':844})
                page.screenshot(path='/tmp/ai6127-fewshot-web-mobile.png',full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                assert not errors,errors
                browser.close()
        finally:
            httpd.shutdown();httpd.server_close();worker.join()
    print('Real Chromium: split selection, four one-call arms, progress, ROUGE, ZIP, offline rescoring, secret exclusion and mobile layout passed.')


if __name__=='__main__':main()
