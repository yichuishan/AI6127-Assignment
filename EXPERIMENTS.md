# 生成实验运行指南

本脚本负责生成模块的实验，不会自动发布或修改检索实现。

## 1. 已准备的真实数据

`experiments/steam_generation_eval.jsonl`：来自指定 GitHub commit
`34e8675a6af11e6871119d2de264a2487ba9b7b5` 的 100 个 query，每条使用
`hybrid_rerank` 已保存排名的前 10 条评论。包含原始 ID、评论、来源链接、游戏、问题类型和相关性标签。

重要限制：上游 `eval/results/runs.json` 是删除未标注评论后的 condensed 排名，不能代表线上 top-10。
相关性等级由人工方面/情感标签按上游规则推导，不是直接人工 query–document 判断。
这些输入适合固定证据的生成消融，报告必须交代这一点。真实端到端评测需从 UI 导出完整检索结果替换 results。

上游有 1,000 条确认过的方面标注，但没有独立标注者日志，因此不能推导 80% 标注一致性。
当前 reference=null、key_points=[]，不会将方面标签自动编造成“人工总结”。

如需重新准备其他检索方案，下载指定提交后运行：

```bash
python run_generation_experiments.py prepare \
  --retrieval-repo /path/to/RAG_ASPA_Game_Retrieval \
  --retrievers bm25 dense hybrid_rrf hybrid_rerank \
  --top-k 10 --output experiments/multi_retriever.jsonl
```

prepare 会拒绝覆盖已有标注文件。`--retrievers` 必须是 runs.json 中的键。
建议先固定 hybrid_rerank 测生成方法，再单独比较检索器，避免一次跑 16×4 组。

## 2. 不发送 API Key 到聊天

本地运行，或在当前环境可访问的交互终端中输入（不会回显，也不会出现在命令历史）：

```bash
read -rsp 'API key: ' GENERATION_API_KEY
export GENERATION_API_KEY
printf '\n'
```

然后在同一个 shell 中执行脚本。终端中的 export 不会自动传到另一个 Codex 工具进程。
如需让 Codex 后续启动的进程读取，请通过环境设置提供 `GENERATION_API_KEY`，
或在终端将密钥写到本仓库以外、权限为 600 的文件（例如 `/tmp/generation-api-key`），
然后使用 `--api-key-file /tmp/generation-api-key`。无需将值发到聊天。
密钥文件不能放入仓库或打包材料，实验结束后自行删除。

支持 `GENERATION_API_KEY` / `OPENAI_API_KEY`；评估模型可单独用 `JUDGE_API_KEY` 或 `--judge-key-file`。
`--base-url` / `--judge-base-url` 指向 OpenAI Chat Completions 兼容端点。
使用当前环境时，需要环境网络设置允许 API 实际域名；仅设置 key 不能改变出网策略。
2026-11-04 再次检查，网站和 /api/annotations 均返回 HTTP 200，1,000 条标注与指定 GitHub 版本完全一致。记录见 experiments/dataset_access.json。当前未配置模型密钥或实际执行付费调用。
不要在 base URL 中包含密码或密钥。

## 3. 不调用 API 的预检查

```bash
python run_generation_experiments.py run experiments/steam_generation_eval.jsonl \
  --output /tmp/generation-preflight --preflight
python -m unittest discover -s tests -v
```

预检查导出 readiness.json、preflight.json、manifest.json，显示缺失的人工参考和覆盖标注。
100 个问题×4 组=400 个输出，默认每条最多 6 个观点，生成阶段上限 2,800 次调用
（共享草稿通常减少实际次数）；启用 judge 额外最多 400 次。先试跑少量问题。

## 4. 小规模真实运行

```bash
python run_generation_experiments.py run experiments/steam_generation_eval.jsonl \
  --output experiments/smoke --limit 3 --model gpt-4o-mini \
  --judge-model YOUR_INDEPENDENT_JUDGE_MODEL --max-claims 4
```

模型名称须替换为你的供应商支持的 ID。生成和评估端点都需要支持 JSON object mode、temperature=0、
max_tokens；当前 transport 输出上限 1800 tokens。模型拒绝、超时、格式错误会记失败，
不会返回伪造成功，也不会自动付费重试。

先检查 checkpoint.jsonl 的 status、error_type 和 result/claim_audit，再决定扩大规模。
失败只保存异常类型，避免异常正文泄漏凭据。failed 记录也会跳过；重试失败项请从输入中筛选
对应 id 并使用新输出目录。若任务在写入 checkpoint 前中断，该任务可能已计费，续跑可能再次调用。

## 5. 正式实验和人工参考

人工逐条阅读该 query 实际 results 中的证据，填写 JSONL 的：

- `reference`：只根据这些证据撰写的总结。
- `key_points`：人工作出的、相关且有证据支持的信息点列表。
- 如有独立 query 相关性标注，可更新 `relevance_grades`（未判断的 chunk 不要填写为 0）。

确保参考可被生成器实际使用的截断后上下文支持；修改上下文预算时重新审核。
必须区分人工参考、人工复核过的机器草稿与纯机器生成参考，并在报告说明来源。

```bash
python -m pip install -r requirements-generation-eval.txt
python run_generation_experiments.py run experiments/steam_generation_eval_labeled.jsonl \
  --output experiments/final --model YOUR_GENERATOR \
  --judge-model YOUR_INDEPENDENT_JUDGE_MODEL \
  --reference-metrics rouge bertscore --repeats 3 \
  --input-price INPUT_USD_PER_MILLION --output-price OUTPUT_USD_PER_MILLION \
  --judge-input-price JUDGE_INPUT_USD_PER_MILLION \
  --judge-output-price JUDGE_OUTPUT_USD_PER_MILLION
```

价格参数需要填数字，按供应商当前价格设置；未设置或 usage 不完整时成本为 null。
价格估算不含缓存折扣等特殊收费。失败调用可能产生未记录账单；最终费用以供应商为准。
BERTScore 默认 roberta-large/CPU，首次需要下载权重；可以 `--device cuda`。
ROUGE 使用英文 stemming；本数据为英文评论，换中文数据应同步调整指标分词和模型。
BERTScorer 在一次运行中复用，不会每条重新加载模型。

## 6. 输出与报告指标

| 输出 | 用途 |
|---|---|
| manifest.json | 输入、Prompt、代码 SHA256，模型与实验参数；不包含密钥文件路径或值 |
| checkpoint.jsonl | 每个 query×repeat×arm 原始输出、检查/修订审计、judge、失败类型 |
| generation_results.json | 完整结果快照 |
| summary.csv | 各组指标均值、有效数量、完成/失败数量、成本/token 总量、P50/P95、串行吞吐 |
| report_data.json | 配对差值和按 query 聚类 bootstrap 95% 区间；问题类型数量 |
| report.md | 报告用主表草稿与限制说明 |
| human_claim_audit.csv | 人工逐观点复核表；提供给两名标注者独立标注 |
| readiness.json | 人工参考、关键点、judge、检索延迟等数据准备情况 |

可用指标包括：

- ROUGE-1/2/L F1、BERTScore F1：仅在有 reference 且指定相应参数时计算。
- Faithfulness、Answer Relevance：独立 judge 的 1–5 分及理由，不是 RAGAS 分数或真值。
- Claim faithfulness fraction：judge 判定完全由所引证据支持的观点比例；空答案为 null。
- Citation precision：judge 判定所引来源支持对应观点的 claim–source 对比例；空引用为 null。
- Context relevance fraction：已标注相关性中 grade>0 的比例，同时报告判断覆盖率。
- Context ranked precision AP：相关结果位置处 precision 的平均；无相关结果为 0，全未标注为 null。
  未判断条目按 condensed judged list 计算，必须结合 judgment coverage 解读。
- Judge context relevance fraction：judge 对每个实际输入 chunk 的直接有用性判断比例。
- Key-point coverage：judge 判断人工关键点是否保留；没有人工关键点时不计算。
- 拒答率、观点数、修订数、删除数、来源校验拒绝数、模型调用数。
- 生成逻辑延迟及 P50/P95、串行 records/s、输入输出 tokens、生成/评估成本分开记录。

缺失值保持 null/空白，不按 0 填充；所有均值都提供有效样本数。
汇总质量指标默认限 status=completed 的样本，失败数量单列；比较时也要分析失败偏差。
任务的生成已成功、judge 失败时，checkpoint 保留生成结果，但该行不会进入完成组均值。
吞吐为串行推理的估计，不代表并发服务负载测试或包含检索/UI 的端到端吞吐。

## 7. 消融的控制方式

A=Baseline；B=结构化+精确摘录；C=B+检查后删除；D=B+修订后复检。
同一 query/repeat 的 B/C/D 从磁盘共用完全相同的模型草稿。不同 repeat 重新生成，
temperature=0 仍不能保证完全确定。任务顺序用 seed 随机打乱。
为了公平，每组延迟/token/成本包含一次逻辑草稿开销；磁盘缓存会减少实际 API 支付。
`measured_execution_seconds` 保留缓存执行墙钟时间，不能用于直接对比三个方法。
因此 summary 中各组成本总和是“分别部署各组”的估算，不等于本次实际账单。

默认所有模型阶段沿用相同模型；独立 judge 只用于最终评估。
相同参数/代码/数据可在同一输出目录续跑；变更实验配置会拒绝混合结果。
配对 bootstrap 先对同一 query 的重复运行求均值，再重采样 query；区间只描述当前数据，
少量 query 无法建立统计显著性。A/B 同时改变提示与摘录约束，不能把增益只归因于其中一项。

## 8. 人工一致性

向至少两名标注者提供互不共享答案的表格，最终整理为：

```csv
record_id,annotator,label
claim_001,A,supported
claim_001,B,supported
claim_002,A,unsupported
claim_002,B,partially_supported
```

```bash
python run_generation_experiments.py agreement independent_labels.csv \
  --output experiments/agreement.json
```

输出每对标注者的原始一致率、Cohen kappa、重叠记录数和双标记录数。
label 应是预先定义的离散类别，不能直接比较自由文本摘要是否字面一致。
多个方面或任务请使用独立 item ID 并分任务报告，避免把标签数量说成独立评论数量。
该命令不会用 Confirmed 状态推算一致性，也不会代替至少 1,000 条人工标注要求。


## 9. 网页实验台（无需在命令中填写密钥）

本地电脑可用 Python 3.10+ 执行 `python launch_web.py`，自动打开浏览器。
基础生成与网页功能无需安装第三方依赖。保持启动终端开启。

```bash
python experiment_web.py --port 8080
```

在运行该命令的电脑浏览器打开 `http://127.0.0.1:8080`。
如果命令在云端执行，需要使用环境提供的端口预览/转发入口；你电脑的 localhost 不会自动指向云端。
仅在需要环境端口预览时使用 `--host 0.0.0.0`。此处启动是当前草稿内的服务，不代表发布了公开网站。

页面支持配置生成 API 地址、模型、API Key，独立最终评估模型、方法、样本数、重复次数、价格及 JSONL 上传。
点击开始才会发送模型请求；默认先跑 3 个问题。可直接填写 `/v1` 基础地址或完整 `/chat/completions` 地址。
进度包括已完成/总任务、当前生成或核验阶段、API 请求数、耗时、失败数和最近输出。
连续 3 项失败会自动停止，避免错误配置下继续执行全部任务。
停止按钮会等待当前请求结束（超时 60 秒），然后导出已完成部分。关闭网页不会停止运行，使用同一浏览器回到页面可恢复显示。
浏览器会话 cookie 只用于区分任务访问；API Key 不进入 cookie、localStorage、服务日志或输出文件。
密钥在运行进程内存中用于请求你填写的服务；请确认填写的服务域名可信。
输出保存在 `experiments/web_runs/<任务ID>/`，网页提供 ZIP 下载。

网页当前不会在服务重启后自动恢复正在执行的任务，也不会保存密钥；已落盘的结果仍可在该目录查看。
ROUGE/BERTScore 仍需要安装可选依赖和提供人工 reference；没有这些条件时可先不勾选。
本页面是当前环境的实验工具；若用于多人公开部署，还需配置正式登录、TLS 与运行资源管理。
