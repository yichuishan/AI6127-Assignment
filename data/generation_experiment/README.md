# Few-shot and Context-compression Experiment

`split.json` freezes a seed-42 stratified question split. The development set has 8 single-aspect, 7 aspect-sentiment, 3 multi-aspect and 2 cross-game questions. The other 80 questions form the test set.

`few_shot_examples.json` contains two demonstrations from the development set: q06 (synthesizing shared benefits and a limitation) and q24 (technical symptoms, conditions and contrasting experiences). Their answer text matches the finalized human-reviewed English references; source texts are exact original passages. Only supporting passages are included to keep demonstrations concise. These two IDs are excluded from scored runs in both partitions, leaving 18 runnable development questions.

The baseline and compression-only methods receive no demonstrations. Few-shot and combined methods receive exactly the same two examples; current test questions and their reference answers never appear in demonstrations. The split and examples are saved, with SHA-256 hashes, in every experiment export.

The split is by question, not by game or review. Reviews and games can occur in multiple questions. The reviewed targets stay unchanged during this experiment. The remaining development questions are available for selecting prompts and compression budgets; the test set is for the final fixed-configuration comparison.

The default compression ratio of 0.6 is an initial setting, not the result of model-based tuning. No paid model experiment was used to select it. The compressor uses only the current question and retrieved text; it does not receive reference answers, relevance judgments or aspect labels.
