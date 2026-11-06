# 生成评测的结构化事实标注（v1）

本目录覆盖指定检索仓库版本中的 **全部 1,000 条已确认评论片段**，并为当前生成实验的
**100 个问题及其固定 top-10 上下文**建立可按 ID 匹配的评测标注。

来源：`GSrihariS/RAG_ASPA_Game_Retrieval`，提交
`34e8675a6af11e6871119d2de264a2487ba9b7b5`。
原始标签文件：`data/steam_annotations_confirmed.json`；问题定义：`eval/queries.jsonl`。
输入上下文：本仓库 `experiments/steam_generation_eval.jsonl`。

## 文件

| 文件 | 内容 |
|---|---|
| `v1/passages.jsonl` | 1,000 条原文、原有人工方面标签、对应事实 ID 和条件候选 ID |
| `v1/facts.jsonl` | 2,898 条“评论—方面—情感”事实，含主观性、原始证据和原文偏移 |
| `v1/query_annotations.jsonl` | 100 个问题的上下文事实池、参考事实、覆盖单元与分歧候选 |
| `v1/conditions_candidates.jsonl` | 409 条硬件、平台、模式、时间/版本、频率候选 |
| `v1/contrast_candidates.jsonl` | 39 个同游戏同方面的正负体验候选组 |
| `v1/fact_review.csv` | 事实与证据的人工复核表 |
| `v1/condition_review.csv` | 条件是否真实、重要、适用于哪个事实的复核表 |
| `v1/contrast_review.csv` | 正负体验是否应在该问题中同时表达的复核表 |
| `v1/statistics.json` | 完整数量与标签分布 |
| `v1/manifest.json` | 原始文件与构建脚本 SHA256、来源提交、生成方法 |

CSV 使用 UTF-8 BOM，可用 Excel 打开；摘录中的换行以 \n / \r 显示，原文仍完整保存在 JSONL。JSONL 使用 UTF-8，每行一个对象。

## “事实”如何切分

每条评论中，每个被提及的方面生成一条事实；`Not mentioned` 不生成事实。
同一方面的 Mixed 标签保持一条，不擅自拆成两个新的人工正负标签。
例如：

```json
{
  "fact_id": "F_a001_graphics_visuals",
  "chunk_id": "steam:1272080:236066609:chunk:001",
  "game_name": "PAYDAY 3",
  "aspect": "graphics_visuals",
  "sentiment": "Positive",
  "subjectivity": "Opinionated",
  "original_support_quote": "The updated visuals look amazing."
}
```

事实 ID 由原始 sample_id 和方面名组成，不随文件排序变化。
原始标签由小组人工审核；将它们转换为事实 ID、规范描述和覆盖单元是本次程序派生。
**本次没有调用 LLM API，也没有把程序派生结果标成新的人工审核 GT。**

这些事实的粒度是“该评论对某方面的情感”，不是对所有自然语言命题的精细拆解。
例如“性能负面”不能自动区分卡顿、低帧率和加载慢；具体描述保留在原文和支持摘录中。

## 支持摘录

2,492 条事实附有可以精确定位的原始支持摘录；406 条原标注未提供摘录。
原标注和原文始终保留；缺失摘录不由模型补造。
`support_span.start/end` 是相对于 `passages.jsonl` 中 text 的 Python Unicode 字符偏移，end 不包含在内。
构建器还支持仅空白差异的可追溯对齐，保证输出 quote 等于原文的 `[start:end]`。

## 问题参考集合与覆盖单元

`context_fact_ids` 是该问题当前十条证据中全部事实的 ID。
`reference_fact_ids` 按问题定义中的目标游戏、方面和情感匹配：

- 指定游戏时只匹配该游戏；跨游戏问题允许任意游戏。
- 只匹配 target_aspects 中的方面。
- target_sentiment=any 时保留各情感类别；指定 Positive/Negative 时严格匹配该类别。

这与检索 qrels 的 0/1/2 规则用途不同：检索可将 Mixed/Neutral 视为部分相关，
本参考集合不把它们当作“正面事实”或“负面事实”。

若多条评论都是“同一游戏—同一方面—同一情感”，它们构成一个 coverage unit。
总结选择其中任何一条事实，即覆盖该标签级单元。共 **227 个 query-specific 覆盖单元**。
这样不会要求总结为每条表达相同标签的评论都单独写一句。

这些参考集合是由目标条件和人工标签推导的，不是人工逐问题判断重要性后的参考摘要。
`human_keypoint_reviewed=false`、`human_summary_reference=null` 明确保留这一来源区别。
它们是评价标签级事实选择的目标，不应包装成细粒度人工摘要关键点。

q76 的当前证据没有满足目标条件的事实。这个样本适合检查空选择/拒答，
不将其 Recall 定义为 100%。所有标注文件仅用于评测，不要把参考 ID 或正确标签泄漏给待测模型。

## 条件与分歧候选

条件候选通过确定性表达式抽取原文明确出现的词组，保留原文位置。
409 条候选中，185 条出现在已有方面支持摘录内并建立了初步事实关联。
其余保留为评论级候选，不强行分配给某一方面。
所有候选 `human_reviewed=false`、`importance=null`。
“online”或“after the update”等词出现，不自动证明它是结论成立的必要条件。
请在 `condition_review.csv` 中审核适用事实和重要性，之后再制作条件保留率 GT。

分歧候选按同一上下文、游戏、方面中的 Positive 与 Negative 标签配对。
共39组；其中25组位于不限定单一情感的问题中，分布在22个问题上。
其余针对特定情感的问题不要求同时表达双方观点。
它们代表不同评论体验，并不自动构成逻辑矛盾：硬件、版本和模式可能不同。
Mixed 标签不会被自动当成正负双方各一条。

## 非模型算法评分

预测文件格式：

```json
{"id":"hybrid_rerank/q01","approach":"A","repeat":0,"selected_fact_ids":["模型选出的事实ID"],"summary":"可选的自然语言总结"}
```

使用：

```bash
python evaluate_fact_selection.py predictions.jsonl --output fact_selection_results.json
```

不调用 LLM 或 embedding 模型，按集合匹配计算：

- `fact_selection_precision`：选择事实中属于 reference_fact_ids 的比例。
- `fact_selection_recall`：参考事实中被选择的比例。
- `fact_selection_f1`：上述 P/R 的调和平均。
- `coverage_unit_recall`：覆盖的游戏/方面/情感单元比例，允许融合重复评论。
- `polarity_contrast_coverage`：适用分歧组中，至少选择一条正面与一条负面事实的比例。
- `out_of_context_fact_rate`：选中的 ID 不属于当前输入事实池的比例。
- 重复 ID 数量、空参考时的正确空选择、各指标有效样本数和数据集覆盖率。

重复 ID 不重复计分。没有适用分歧组时不计算分歧比例。
参考和输出均为空时，P/R/F1 为 null，单独记录正确空选择。
条件尚未复核，`condition_retention_rate` 保持 null，不将候选直接当成 GT。

**这些分数评价结构化 ID 选择，不证明 summary 的自由文本也表达正确。**
旧版 generation_results.json 没有 selected_fact_ids，不能直接套用此评分器；
当前新版网页已让每条 claim 输出 fact_ids，并从最终观点合并为 selected_fact_ids。旧实验仍需人工映射或重新生成。不要用标准答案反填预测后报告实验分数。

## 重新构建

```bash
python annotate_generation_facts.py \
  --annotations /path/to/RAG_ASPA_Game_Retrieval/data/steam_annotations_confirmed.json \
  --queries /path/to/RAG_ASPA_Game_Retrieval/eval/queries.jsonl \
  --generation-cases experiments/steam_generation_eval.jsonl \
  --output data/generation_annotations/v2
```

构建器拒绝覆盖非空输出目录，原始人工标签和此前实验结果不会被改写。


## 网页已接入

`python launch_web.py` 使用新事实选择流程，不调用独立评测模型。
为防止从候选列表泄漏方面存在性，每条评论的八个方面均生成输入槽位，
不会直接把 facts.jsonl 中仅包含已提及方面的列表发给模型。正确标签和参考集只用于评分。
详见 EXPERIMENTS.md §13。
