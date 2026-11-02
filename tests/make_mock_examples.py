"""Regenerate synthetic example outputs without model/API calls."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_generator import generate_summary
from test_verification import HITS, BAD, FIXED, verdict, ScriptedClient


def main():
    sequences = {
        'baseline': [{'claims': [BAD]}],
        'improved': [{'claims': [BAD]}],
        'verify_delete': [{'claims': [BAD]}, verdict('partially_supported')],
        'verify_revise': [{'claims': [BAD]}, verdict('partially_supported'), {'claim': FIXED}, verdict('supported')],
    }
    examples = []
    for approach, outputs in sequences.items():
        examples.append({
            'provenance': 'SCRIPTED MOCK: synthetic evidence, preset draft/verdict/repair. No LLM called; timings are local mock execution only.',
            'result': generate_summary('How is performance?', HITS, approach=approach,
                                       model='offline-mock-not-a-real-model', client=ScriptedClient(*outputs))})
    examples.append({'provenance': 'Actual no-evidence code path. No model invoked.',
                     'result': generate_summary('How is performance?', [])})
    target = Path(__file__).resolve().parents[1] / 'generation_results.json'
    target.write_text(json.dumps({'synthetic_input_hits': HITS, 'examples': examples}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
