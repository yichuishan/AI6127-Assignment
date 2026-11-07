"""Verify the exported whitelist runs without falling back to the source checkout."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from export_generation import CODE_FILES, DATA_DIR, DATA_FILES, MODULE_DIR, export

SMOKE = """
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
module = root/'code/generation/rag_summary'
sys.path.insert(0, str(module))
from run_generation_experiments import BENCHMARK, jsonl, run
from generation_experiment_design import select_cases
from evaluate_generation import rescore
from generation_paths import RESULTS_DIR
class Model:
    def complete(self, messages, model, temperature):
        payload = json.loads(messages[1]['content'])
        evidence = payload['evidence'][0]
        return {'content': json.dumps({'claims': [{'text': 'Players describe their experience.',
                'chunk_ids': [evidence['chunk_id']]}]}),
                'usage': {'prompt_tokens': 20, 'completion_tokens': 10}}
cases = select_cases(jsonl(BENCHMARK))[:1]
rows = run(cases, RESULTS_DIR/'smoke', client=Model(), model='offline',
           concurrency=2, as_of='2026-11-07T00:00:00Z')
assert len(rows) == 4 and all(r['status'] == 'completed' for r in rows)
assert {r['approach'] for r in rows} == {'baseline', 'few_shot', 'compressed', 'combined'}
assert rescore(rows) == rows
# The public API used by the group app must also work through its package import.
sys.path.insert(0, str(root/'code'))
from generation.rag_summary.rag_generator import generate_summary
answer = generate_summary(cases[0]['query'], cases[0]['results'], client=Model(),
                          model='offline', compression_ratio=.75)
assert answer['status'] == 'ok' and answer['citations']
assert Path(sys.modules['run_generation_experiments'].__file__).parent == module
print('Four methods, offline ROUGE rescore, package API and compression passed.')
"""


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root/'code').mkdir()
        (self.root/'code/paths.py').write_text('# group fixture\n')
        self.env = dict(os.environ)
        for name in ('RAG_SUMMARY_DATA_DIR', 'RAG_SUMMARY_RESULTS_DIR'):
            self.env.pop(name, None)

    def test_export_has_exact_runtime_whitelist_and_runs_in_isolation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            module, data = export(self.root)
        self.assertEqual({str(p.relative_to(module)) for p in module.rglob('*') if p.is_file()}, set(CODE_FILES))
        self.assertEqual({str(p.relative_to(data)) for p in data.rglob('*') if p.is_file()}, set(DATA_FILES.values()))
        result = subprocess.run([sys.executable, '-I', '-B', '-c', SMOKE, str(self.root)],
                                cwd=self.root, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        for script in ('rag_generator.py', 'run_generation_experiments.py',
                       'evaluate_generation.py', 'experiment_web.py'):
            result = subprocess.run([sys.executable, '-B', str(module/script), '--help'],
                                    cwd=self.root, env=self.env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_replace_removes_old_code_only_and_keeps_backup_outside_group(self):
        module = self.root/MODULE_DIR
        (module/'tests').mkdir(parents=True)
        (module/'tests/old.py').write_text('old')
        sibling = module.parent/'aspect_sentiment.py'
        sibling.write_text('other contribution')
        data = self.root/DATA_DIR
        (data/'rag_summary').mkdir(parents=True)
        (data/'rag_summary/steam_generation_eval.jsonl').write_text('previous benchmark')
        (data/'queries.jsonl').write_text('retriever dataset')
        with self.assertRaises(ValueError):
            export(self.root)
        output = io.StringIO()
        backup = None
        try:
            with contextlib.redirect_stdout(output):
                export(self.root, replace=True)
            line = next(line for line in output.getvalue().splitlines() if line.startswith('Previous files'))
            backup = Path(line.split(': ', 1)[1])
            self.assertFalse(backup.is_relative_to(self.root))
            self.assertEqual((backup/MODULE_DIR/'tests/old.py').read_text(), 'old')
            self.assertFalse((module/'tests').exists())
            self.assertEqual(sibling.read_text(), 'other contribution')
            self.assertTrue((data/'rag_summary_benchmark.jsonl').is_file())
            self.assertEqual((data/'queries.jsonl').read_text(), 'retriever dataset')
            self.assertFalse((data/'rag_summary').exists())
            self.assertEqual((backup/DATA_DIR/'rag_summary/steam_generation_eval.jsonl').read_text(),
                             'previous benchmark')
        finally:
            if backup:
                shutil.rmtree(backup)


if __name__ == '__main__':
    unittest.main()
