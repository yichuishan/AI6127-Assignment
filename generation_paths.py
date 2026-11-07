"""Resolve standalone and group-project data locations without changing the API."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Group layout: <project>/code/generation/rag_summary/*.py
GROUP_ROOT = (ROOT.parents[2] if ROOT.name == 'rag_summary'
              and ROOT.parent.name == 'generation' and ROOT.parents[1].name == 'code'
              else None)

_data = os.getenv('RAG_SUMMARY_DATA_DIR')
DATA_DIR = (Path(_data).expanduser().resolve() if _data else
            GROUP_ROOT/'data/evaluation_dataset/rag_summary' if GROUP_ROOT else ROOT/'data')
BENCHMARK = (DATA_DIR/'steam_generation_eval.jsonl' if _data or GROUP_ROOT
             else ROOT/'experiments/steam_generation_eval.jsonl')
DESIGN_DIR = DATA_DIR/'generation_experiment'
DRAFTS = DATA_DIR/'reference_summaries/drafts/reference_drafts.jsonl'
REFERENCES = DATA_DIR/'reference_summaries/reviewed/approved_references.jsonl'
_results = os.getenv('RAG_SUMMARY_RESULTS_DIR')
RESULTS_DIR = (Path(_results).expanduser().resolve() if _results else
               GROUP_ROOT/'data/generation_results/rag_summary' if GROUP_ROOT
               else ROOT/'experiments/web_runs')
