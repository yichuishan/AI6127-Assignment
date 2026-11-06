# AI6127: Evidence-grounded Review Summarization

最终实验代码：固定检索证据，比较四种 RAG 总结方法，并使用人工复核后的事实 ID 参考集合进行算法评测。

需要 **Python 3.10+**。全部使用标准库，无须安装评测模型或第三方 Python 包。

## 网页运行

```bash
python launch_web.py
```

浏览器打开后填写生成模型的 API 地址、模型名称和 API Key，设置问题数量、重复次数及最多观点数，即可查看进度并下载结果 ZIP。

默认配置为 `https://api.deepseek.com`、`deepseek-v4-pro`，自动关闭官方 DeepSeek V4 的思考模式。默认先运行 3 个问题；完整实验选择 100 个问题、重复 1 次、最多 4 条观点。

API Key 仅保存在进程内存中，不写入结果文件或浏览器存储。网页运行期间保留终端；按 Ctrl+C 停止服务。

## 四种生成方法

| 方法 | 流程 |
|---|---|
| A · Baseline | 生成观点、来源引用及事实 ID |
| B · Evidence quotes | 在 A 的基础上强化证据提示，要求并校验原文摘录 |
| C · Verify-delete | 使用 B 的初稿，逐观点自检，删除未通过的观点 |
| D · Verify-revise | 使用 B 的初稿，修订未通过的观点并再次核查 |

B/C/D 共用初始草稿。C/D 的自检使用生成模型；最终评分由同一个算法完成，不配置独立评测模型。

生成接口位于 `rag_generator.py`：

```python
from rag_generator import generate_summary

result = generate_summary(
    question,
    ranked_passages,
    approach="improved",
    model="deepseek-v4-pro",
    max_claims=4,
)
```

`ranked_passages` 中每项需要 `chunk_id`、`text`，可在 `metadata` 中提供 `game_name`、`source_url`。结构化实验由运行器添加每条评论的八个无标签方面候选；模型输入不包含参考答案或人工情感标签。

## 命令行实验

先在自己的终端设置 `GENERATION_API_KEY`，然后运行：

```bash
python run_fact_experiments.py --output experiments/runs/full --limit 100 --repeats 1 --max-claims 4 --base-url https://api.deepseek.com --model deepseek-v4-pro
```

先试跑时把 `--limit 100` 改为 `--limit 3`。输出目录必须是新的空目录。

运行结果包含完整生成输出、逐任务记录、预测 ID、使用的参考集合、指标 CSV、汇总和配置哈希。

## 无 API 重算

重新评分下载的实验 ZIP：

```bash
python recompute_results.py generation-experiment.zip --output experiments/runs/rescored
```

复现报告所用结果，不需要密钥：

```bash
python recompute_results.py experiments/final/generation_results.json --output experiments/runs/report_reproduction
```

仅对预测 JSONL 评分：

```bash
python evaluate_fact_selection.py experiments/final/predictions.jsonl --output experiments/runs/prediction_metrics.json
```

预测文件每行包含 `id`、`selected_fact_ids`，可附带 `approach`、`repeat`、`summary`。脚本使用仓库内的最终参考集合。

## 数据和评分

- `experiments/steam_generation_eval.jsonl`：100 个固定问题及每题至多十条已标注评论证据，来自保存的 `hybrid_rerank` 排名。
- `data/generation_annotations/`：1,000 条原始标注评论、2,898 个评论—方面事实，以及最终参考集合。
- `human_review.json`：40 项用户复核判断，涉及 16 个问题：22 项核心、7 项可选补充、11 项排除。

核心信息计入参考集合；可选补充不强制覆盖，也不按误选扣分；排除信息被选中时计为误选。未复核条目沿用原标注匹配规则。

指标包括事实选择 Precision/Recall/F1、覆盖单元召回率、正负分歧覆盖率，以及运行耗时、调用次数和 Token 用量。覆盖单元按游戏、方面、情感分组。最终参考集合有 591 个必需问题—事实对、238 个覆盖单元，以及 24 个问题中的 27 个正负分歧组。条件保留率尚无对应人工标注，不计分。

这些指标评价结构化事实选择。详细来源、字段和规则见 [数据说明](data/generation_annotations/README.md)。网页仅接受原基准中问题和证据保持一致的子集。

## 最终实验结果

`experiments/final/` 保存报告所用的真实生成输出及复核后指标。得分均为按问题宏平均的百分比：

| 方法 | 精确率 | 召回率 | F1 | 单元覆盖 | 正负覆盖 |
|---|---:|---:|---:|---:|---:|
| A | 85.40 | 72.71 | 75.82 | 88.73 | 56.25 |
| B | 88.03 | 69.68 | 75.12 | 87.18 | 50.00 |
| C | 89.49 | 68.67 | 74.74 | 87.05 | 50.00 |
| D | 89.20 | 69.58 | 75.23 | 87.05 | 50.00 |

A/B 各完成 100 项，C/D 各完成 99 项。逐项输出、有效数量、Token 和时间见 [最终结果 CSV](experiments/final/summary.csv)。

## 文件结构

```text
rag_generator.py            生成、引用校验、自检与修订
prompts/                    A/B 提示词、事实选择、自检及修订提示词
run_fact_experiments.py      固定基准实验运行器
experiment_web.py            网页后端、进度与 ZIP 导出
launch_web.py                本地启动入口
web/                         网页前端
evaluate_fact_selection.py   统一算法评分器
recompute_results.py         已保存输出的无 API 重算
data/generation_annotations/ 最终数据与人工复核记录
experiments/                 固定输入、来源审计和最终实验结果
tests/                       生成、评分、网页及结果复现测试
```

## 验证

```bash
python -m unittest discover -s tests -v
```

测试使用模拟模型和已保存的最终输出，不调用付费 API。
