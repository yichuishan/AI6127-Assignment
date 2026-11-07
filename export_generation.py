"""Export runtime files into the group's code/ and data/ layout; no API key needed."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parent
CODE_FILES = (
    'rag_generator.py', 'review_context.py', 'context_compression.py',
    'experiment_execution.py', 'generation_experiment_design.py',
    'generation_paths.py', 'run_generation_experiments.py',
    'evaluate_generation.py', 'experiment_web.py', 'launch_web.py',
    'prompts/baseline.txt', 'prompts/few_shot.txt',
    'web/index.html', 'web/app.js', 'web/style.css',
)
DATA_FILES = {
    'experiments/steam_generation_eval.jsonl': 'rag_summary_benchmark.jsonl',
    'data/generation_experiment/split.json': 'rag_summary_split.json',
    'data/generation_experiment/few_shot_examples.json': 'rag_summary_few_shot_examples.json',
    'data/reference_summaries/drafts/reference_drafts.jsonl': 'rag_summary_reference_contexts.jsonl',
    'data/reference_summaries/reviewed/approved_references.jsonl': 'rag_summary_approved_references.jsonl',
}
MODULE_DIR = Path('code/generation/rag_summary')
DATA_DIR = Path('data/evaluation_dataset')
# Migrate only this exporter's five previous data paths, preserving other files.
LEGACY_FILES = (
    'steam_generation_eval.jsonl',
    'generation_experiment/split.json',
    'generation_experiment/few_shot_examples.json',
    'reference_summaries/drafts/reference_drafts.jsonl',
    'reference_summaries/reviewed/approved_references.jsonl',
)
CHECK = """
import sys
from pathlib import Path
module = Path(sys.argv[1])
sys.path.insert(0, str(module))
from run_generation_experiments import BENCHMARK, benchmark, jsonl
from generation_experiment_design import load_examples, load_split
from generation_paths import DATA_DIR, RESULTS_DIR
cases = jsonl(BENCHMARK)
refs = benchmark(cases)
examples = load_examples(cases, refs)
split = load_split()
assert len(cases) == len(refs) == 100 and len(examples) == 2
assert len(split['test_ids']) == 80
assert DATA_DIR == module.parents[2]/'data/evaluation_dataset'
assert RESULTS_DIR == module.parents[2]/'data/generation_results/rag_summary'
import experiment_web
sys.path.insert(0, str(module.parents[1]))
from generation.rag_summary.rag_generator import generate_summary
assert callable(generate_summary)
print('Validated: runtime imports, 100 fixed contexts/references, 20/80 split, 2 demonstrations.')
"""


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def export(group_root, replace=False):
    root = Path(group_root).expanduser().resolve()
    if not (root/'code/paths.py').is_file():
        raise ValueError('--group-root must contain the group project\'s code/paths.py')
    module, data = root/MODULE_DIR, root/DATA_DIR
    if module.is_symlink() or data.is_symlink():
        raise ValueError('The module and data destinations must not be symbolic links')
    if module.exists() and not replace:
        raise ValueError(f'{module} already exists; use --replace to back it up and replace only this module')
    for name in (*CODE_FILES, *DATA_FILES):
        if not (SOURCE/name).is_file():
            raise ValueError(f'Missing required source file: {name}')

    # Validate an isolated, complete export before changing the group repository.
    with tempfile.TemporaryDirectory(prefix='rag-summary-stage-') as temporary:
        stage = Path(temporary)
        for name in CODE_FILES:
            copy_file(SOURCE/name, stage/MODULE_DIR/name)
        for name, target in DATA_FILES.items():
            copy_file(SOURCE/name, stage/DATA_DIR/target)
        import os
        env = dict(os.environ)
        for name in ('RAG_SUMMARY_DATA_DIR', 'RAG_SUMMARY_RESULTS_DIR'):
            env.pop(name, None)
        subprocess.run([sys.executable, '-I', '-B', '-c', CHECK, str(stage/MODULE_DIR)],
                       check=True, env=env, cwd=stage)

        legacy = [data/'rag_summary'/name for name in LEGACY_FILES]
        old_files = [path for path in [*(data/name for name in DATA_FILES.values()), *legacy]
                     if path.is_file()]
        backup = None
        if module.exists() or old_files:
            backup = Path(tempfile.mkdtemp(prefix='rag-summary-backup-'))
            if module.exists():
                shutil.copytree(module, backup/MODULE_DIR)
            for old in old_files:
                copy_file(old, backup/old.relative_to(root))
            print(f'Previous files backed up outside the repository: {backup}')
        if module.exists():
            shutil.rmtree(module)
        shutil.copytree(stage/MODULE_DIR, module)
        for target in DATA_FILES.values():
            copy_file(stage/DATA_DIR/target, data/target)
        for old in legacy:
            if old.is_file():
                old.unlink()
                parent = old.parent
                while parent != data:
                    try:
                        parent.rmdir()
                    except OSError:
                        break
                    parent = parent.parent
    print(f'Runtime: {module} (10 Python files, 2 prompts, 3 web assets)')
    print(f'Required data: {data} (5 files)')
    print('No tests, docs, app.py, API keys or saved experiment outputs copied.')
    return module, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group-root', required=True, help='Group repository root containing code/paths.py')
    parser.add_argument('--replace', action='store_true', help='Back up and replace an existing rag_summary module')
    args = parser.parse_args()
    try:
        export(args.group_root, args.replace)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Export failed: {error}\n')


if __name__ == '__main__':
    main()
