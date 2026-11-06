# Structured fact selection results

Scoring uses exact ID matching; no evaluator model is called.
All eight aspect slots per review are visible to generation; gold labels and reference sets are not.
P/R/F1 concern review-aspect selection. Natural-language summary entailment is not scored.
References include 40 query-specific human decisions. Optional background is not required or penalized.
Condition retention has not been annotated and is not scored.
C/D self-verification uses the generator model and is part of generation, not the final scorer.

| Method | Completed | Failed | Precision | Recall | F1 | Unit recall | Contrast coverage | Latency s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| improved | 100 | 0 | 0.880297619047619 | 0.6968075134741801 | 0.7512300445174073 | 0.8717652717652717 | 0.5 | 4.806933009999999 |
| verify_delete | 99 | 1 | 0.8948615160349854 | 0.6867166280431587 | 0.7473789992713351 | 0.8704567541302235 | 0.5 | 11.764731568686875 |
| baseline | 100 | 0 | 0.8540357142857142 | 0.7271233816688362 | 0.7581800018684342 | 0.8873496873496873 | 0.5625 | 3.1240646 |
| verify_revise | 99 | 1 | 0.8919703595724003 | 0.6957545289177942 | 0.7523277919752299 | 0.8704567541302235 | 0.5 | 12.042440570707072 |

All metric means disclose valid counts in summary.csv. Failed tasks are listed separately.
B/C/D share a draft; logical latency, usage and cost include one draft per arm.

## Paired F1 comparisons

| After - before | Queries | Delta (pp) | Bootstrap 95% CI (pp) |
|---|---:|---:|---:|
| improved - baseline | 99 | -0.695 | [-3.844, 2.416] |
| verify_delete - improved | 98 | -0.131 | [-0.972, 0.617] |
| verify_revise - improved | 98 | 0.250 | [0.000, 0.641] |
