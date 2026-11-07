"""Optional check: install rouge-score==0.1.2 in a test venv, then run this script."""
import json
from pathlib import Path
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_generation import score_summary, strip_citations


def main():
    from rouge_score import rouge_scorer
    standard = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],use_stemmer=False)
    rng = random.Random(42)
    words = ['game','story','story','combat','runs','smoothly','crashes','RTX-4060',"isn't",'Époque','50–60','PC!']
    pairs = [('', ''),('good','good'),('a a a b','a a b'),('a c b','a b c'),
             ('Good story. [steam:1:2:chunk:001]','good story'),('Beautiful Époque visuals','belle époque visuals')]
    for _ in range(500):
        pairs.append((' '.join(rng.choices(words,k=rng.randrange(35))),
                      ' '.join(rng.choices(words,k=rng.randrange(35)))))
    largest = 0
    for prediction,reference in pairs:
        ours = score_summary(prediction,reference)
        expected = standard.score(strip_citations(reference),strip_citations(prediction))
        for metric,score in expected.items():
            for suffix,value in [('precision',score.precision),('recall',score.recall),('f1',score.fmeasure)]:
                delta = abs(ours[f'{metric}_{suffix}']-value)
                largest = max(largest,delta)
                assert delta < 1e-12, (prediction,reference,metric,suffix)
    print(json.dumps({'standard':'google rouge-score 0.1.2','stemming':False,'text_pairs':len(pairs),
                      'metrics_per_pair':9,'max_absolute_error':largest,'seed':42},indent=2))


if __name__=='__main__':main()
