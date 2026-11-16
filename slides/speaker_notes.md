# RAG summarization — speaker notes

Three slides; English narration: 221 words, approximately 1.5–2 minutes.

## Slide 1: Evidence-grounded generation

My part turns the retriever's review passages into a concise answer with traceable citations. We first preprocess the reviews and preserve their source IDs and metadata. We then construct the prompt, call DeepSeek, and validate the returned citation IDs. I tested two improvements: adding two reviewed examples to the prompt, and compressing the context with query-aware sentence selection.

**中文对照**

我负责把检索器返回的评论整理成简洁、带可追溯引用的回答。首先预处理评论，保留来源 ID 和元数据；随后构造提示词，调用 DeepSeek，并校验返回的引用 ID。我测试了两项改进：在提示词中加入两个经过核验的示例，以及根据问题选择相关句子，对上下文进行压缩。

## Slide 2: Query-aware context compression

Context compression works on original sentences. We split the retrieved reviews into sentences and score their relevance to the question with TF-IDF cosine similarity. We then select sentences using eighty percent relevance and a twenty percent penalty for overlap with already selected content. When the budget allows, we retain one neighboring sentence on each side. Finally, we restore the original order and preserve source IDs. In this experiment, the target budget was seventy-five percent of the original characters.

**中文对照**

上下文压缩直接处理原始句子。首先将评论分句，用 TF-IDF 余弦相似度衡量每句话与问题的相关性；然后按 80% 的相关性分数和 20% 的重复惩罚选择句子。当预算允许时，保留前后各一句，帮助保留上下文。最后恢复原始顺序，并保留来源 ID。这次实验的目标预算是原始评论字符数的 75%。

## Slide 3: Experimental results

We compared four settings on eighty fixed test questions. Few-shot produced the best scores. Compared with baseline, ROUGE-2 increased from 8.63 to 10.38, and ROUGE-L increased from 19.05 to 20.54. Both paired confidence intervals were above zero. Compression reduced input tokens by 9.4 percent on matched questions, while keeping ROUGE-L close to baseline. Combining the methods kept most of the few-shot improvement, but did not improve further. The main takeaway is: few-shot helps answer quality, while compression reduces input cost.

**中文对照**

我们在 80 个固定测试问题上比较了四组方法。少样本提示取得了最高分：ROUGE-2 从 8.63 提升到 10.38，ROUGE-L 从 19.05 提升到 20.54，两项配对提升的置信区间都高于零。压缩在配对问题上减少了 9.4% 的输入 Token，同时 ROUGE-L 接近基线。两者结合保留了大部分少样本收益，但没有进一步提高分数。少样本主要改善回答质量，压缩主要减少输入成本。
