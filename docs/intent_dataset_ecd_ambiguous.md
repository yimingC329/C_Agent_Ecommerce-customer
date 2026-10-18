# ECD + Chinese Ambiguous Reference 意图数据集

## 最终结果

项目已在 `data/intent/ecd_ambiguous_4000/` 生成 4000 条去重电商对话意图数据。

| 集合 | 数量 | 用途 |
| --- | ---: | --- |
| `mining.jsonl` | 2400 | 难负样本挖掘、few-shot 三轮迭代 |
| `validation.jsonl` | 800 | 提示词和迭代轮次选择 |
| `test.jsonl` | 800 | 最终锁定评测，不参与样例选择 |

来源各 2000 条：

- [ECD（E-commerce Dialogue Corpus）](https://github.com/cooelf/DeepUtteranceAggregation)：2000 条。
- [Chinese Ambiguous Reference](https://github.com/ygan/Chinese-Ambiguous-Reference)：2000 条。

最终标签分布：

| 标签 | 数量 |
| --- | ---: |
| `general-query` | 592 |
| `additional-query` | 2432 |
| `graphrag-query` | 976 |

这两个来源都是文本对话数据，因此本数据集不伪造 `image-query` 和 `file-query`。
这两个模态标签应继续用项目的图片/文件上传集成测试单独验证，不能合并进本次
文本三分类 F1。

## 标注方法

1. ECD 只读取作者完整训练集中的正例会话，抽取客服回复之前的最后一条顾客话语。
2. Chinese Ambiguous Reference 先利用原始 `split_sentence_type` 标注筛选顾客话语：
   下一客服行为为 2/4 时形成“需补充信息”候选，1/3 时形成“可业务回答”候选。
3. 两个来源最终都按项目当前路由定义，由 `.env` 中配置的 DeepSeek 模型进行
   三分类复标，并写入 `label_reason`。
4. 每条记录保留 `source_dataset`、`source_dialogue_id`、原始行为码或来源说明、
   `label_method` 和上下文，便于后续逐条人工复核。

标签语义与线上路由一致：

- `general-query`：闲聊、寒暄、致谢、确认，或无需查业务知识即可回复。
- `additional-query`：与业务相关但缺少商品、订单、规格或故障信息，必须先追问。
- `graphrag-query`：问题明确，需要查询商品、订单、物流、库存、价格、活动、售后
  或使用知识。

## 防止数据泄漏

脚本使用固定随机种子 `20261007`，先对 `source_dataset + source_dialogue_id` 做
稳定哈希，再按 60% / 20% / 20% 分配到挖掘、验证和测试分区，最后才在各分区内
定额抽样。因此同一段对话产生的多条语句不会跨集合。

实际验证结果：

- 总记录数：4000
- 唯一问题数：4000
- 来源计数：ECD 2000，Chinese Ambiguous Reference 2000
- 对话跨集合泄漏：0

## 重新生成

在 `llm_backend` 目录执行：

```powershell
& '..\..\.venv\Scripts\python.exe' scripts\prepare_ecd_ambiguous_intents.py
```

标注会分别缓存到 `ecd_label_cache.jsonl` 和 `ambiguous_label_cache.jsonl`，中断后
重新运行会从缓存继续，不会重复请求已经完成的样本。

## 验证数据集

```powershell
& '..\..\.venv\Scripts\python.exe' scripts\verify_ecd_ambiguous_dataset.py
```

只有输出 `"status": "PASS"`，且 `dialogue_leakage` 为 0，才能开始难负样本实验。

## 人工复核

用 Excel 打开 `review.csv`，重点复核 `additional-query` 与 `graphrag-query` 的边界：

1. 在 `review_label` 中填修正后的标签；正确时可保持为空。
2. 在 `review_notes` 记录修改原因。
3. 复核时只参考 `query` 与 `context`，不要根据模型给出的理由迁就原标签。
4. 锁定 `test` 分区后，不用测试集反向修改提示词或 few-shot 样例。

数据集统计只能用于当前自动标注版本。完成复核后，应重新导出清洗版并再次运行
验证脚本，之后才能把其 F1 写入简历。

## 上下文复核结果

已使用 `scripts/review_intent_labels.py` 对全部 4000 条记录完成上下文复核，结果保存在
`data/intent/ecd_ambiguous_4000_reviewed/`。复核流程为：

1. 上下文审查器在不知道原标签的情况下独立分类。
2. 边界审查器独立检查 `additional-query` / `graphrag-query`、短问句、确认、
   抱怨与业务咨询的边界。
3. 两次判断不一致、置信度低于 0.8，或建议修改原标签的记录进入第三次裁决。
4. 对价格、库存、短问句和业务抱怨等 556 条高风险边界记录执行最终定向质检。

最终统计：

| 项目 | 数量 |
| --- | ---: |
| 总记录 | 4000 |
| `general-query` | 459 |
| `additional-query` | 814 |
| `graphrag-query` | 2727 |
| 相对初始标签发生变化 | 2211 |
| 进入第三次裁决 | 2421 |
| 进入最终定向质检 | 556 |
| 定向质检再次修改 | 168 |
| 平均复核置信度 | 0.897 |
| 最终置信度低于 0.8 | 23 |
| 唯一 query | 4000 |
| 对话跨集合泄漏 | 0 |

大量 `additional-query → graphrag-query` 修改来自上下文指代恢复。例如，“多少钱？”
单独看缺少对象，但历史已经明确具体商品时，系统下一步应查询价格，而不是重复追问。
明确商品类别的“芒果怎么卖”“有冰箱贴吗”也应查询商品/库存信息；个性化推荐缺少
用途、预算、尺寸或肤质等必要条件时，才归为 `additional-query`。

复核产物：

- `all.jsonl`：4000 条最终标签及完整复核证据。
- `mining.jsonl`、`validation.jsonl`、`test.jsonl`：保持原对话级隔离划分。
- `all_reviewed.csv`：全部复核结果。
- `changed_labels.csv`：仅包含发生修改的 2211 条记录。
- `review_summary.json`：标签分布与转换统计。

重新执行复核：

```powershell
& '..\..\.venv\Scripts\python.exe' scripts\review_intent_labels.py
```

复核缓存位于 `ecd_ambiguous_4000_reviewed/cache/`，脚本可断点续跑。该过程是多次
模型辅助复核，不应在报告中描述为人工逐条标注；最终标签保留了原标签、各复核意见、
裁决原因、置信度和是否修改等可追溯字段。

最终仍有 23 条真实对话因为原文口语化、上下文残缺或指代不完整而低于 0.8；这些记录
没有被静默删除，而是保留最终裁决标签与低置信度，后续难负样本实验可优先把它们作为
高价值边界样本检查。

## 电商路由一致性清洗与重新实验

进一步实验发现 Chinese Ambiguous Reference 同时包含餐饮点单、住宿、线下交易执行和
零售咨询，部分轮次无法稳定映射到本项目的三类知识路由。为避免通过矛盾标签虚增或破坏
指标，使用 `scripts/curate_intent_eval_dataset.py` 对该来源执行两次相互独立的可评测性
审核；只有两个审核器标签一致且最低置信度不低于 0.85 的记录才保留。ECD 保留已有
复核置信度不低于 0.80 的记录。

清洗版位于 `data/intent/ecd_ambiguous_curated/`：

| 项目 | 数量 |
| --- | ---: |
| 输入 | 4000 |
| 保留 | 3063 |
| 排除 | 937 |
| mining | 1832 |
| validation | 617 |
| test（仍锁定） | 614 |
| 双审核分歧排除 | 158 |
| 不适配路由体系排除 | 728 |
| ECD低置信排除 | 51 |
| 对话跨集合泄漏 | 0 |

多轮 few-shot 现在使用两类独立样本：

- `data/intent/canonical_anchors.jsonl`：9 条固定典型锚点，三个文本类别各3条；
- 难负样本：动态预算6条；新样本先经过问题、上下文、双复核一致性和领域质量门控，
  再通过消融实验决定实际替换数量，不再固定采用“3条历史+3条最新”。

难例同时按真实标签和误判方向均衡。餐饮点单、住宿等跨领域数据仍可用于
验证泛化，但不允许进入 few-shot Prompt。每轮还会与上一个已接受检查点比较：
Macro-F1 下降，或任一类 F1 下降超过0.03，该轮就被拒绝并自动回退。

在清洗数据上的实际第一轮结果：

| 轮次 | Accuracy | Macro-F1 | General F1 | Additional F1 | GraphRAG F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Round 0 zero-shot | 0.7439 | 0.7361 | 0.7544 | 0.6649 | 0.7891 |
| Round 1 few-shot（修正审核理由后） | 0.8541 | 0.8478 | 0.8421 | 0.8199 | 0.8812 |
| Round 2（9锚点改造前，已拒绝） | 0.8444 | 0.8430 | 0.8525 | 0.8131 | 0.8636 |

当前已接受的最佳检查点是 Round 1（Macro-F1 0.8478）。改造前的 Round 2 因
Macro-F1 下降0.0047被性能门禁拒绝，其难例已从运行目录回退。完整报告位于
`data/intent/experiments/curated_round2_balanced_gate/intent_optimization_summary.json`。该结果来自
`deepseek-chat` 的模型自报概率，不应描述为真实 token logprobs，也不能预先宣称达到
0.93；后续数字必须来自锁定流程的实际报告。

## 一致标签 v2 与第二轮消融结果（2026-10-07）

`ecd_ambiguous_curated_v2/` 对 mining、validation、锁定 test 使用同一条双复核一致
规则重新裁决，共修正421条标签；validation仍固定为617条。以 v2 第一轮为唯一对照：

| 轮次 | Accuracy | Macro-F1 | General F1 | Additional F1 | GraphRAG F1 | 门控 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Round 1 v2 | 0.7618 | 0.7697 | 0.8851 | 0.6184 | 0.8057 | 已接受 |
| Round 2 自动3例 | 0.7520 | 0.7355 | 0.7962 | 0.6006 | 0.8097 | 已拒绝并回滚 |
| Round 2 人工3例 | 0.7682 | 0.7599 | 0.8482 | 0.6103 | 0.8214 | 已拒绝并回滚 |
| Round 2 单例消融 | 0.7844 | 0.7814 | 0.8778 | 0.6347 | 0.8317 | **已接受** |
| Round 3 Additional单例 | 0.7374 | 0.7532 | 0.8810 | 0.6045 | 0.7731 | 已拒绝并回滚 |
| Round 3 GraphRAG单例 | 0.7747 | 0.7818 | 0.9000 | 0.6307 | 0.8148 | **已接受** |

最终第二轮只新增一条双复核一致的上下文边界样本“自然偏白吧。”，同时保留5条
第一轮难例。相对第一轮，Macro-F1提升0.0117，Additional F1提升0.0163，GraphRAG
F1提升0.0260，General F1下降0.0073（未超过0.03门限），617条验证请求错误数为0。
最终报告位于
`data/intent/experiments/curated_v2_round2_graph_ablation/intent_optimization_summary.json`。

第三轮先用已接受的第二轮检查点在1832条 mining 数据上重新挖掘，55条候选均先以
`pending` 保存。Additional 单例使大量 GraphRAG 样本被过度改判为追问，因而被门控
拒绝；GraphRAG 上下文单例“我需要大容量的。”通过门控。第三轮 Macro-F1 为0.7818，
相对第二轮仅提升0.0004，应解释为基本持平的边界微调，而非显著提升。报告位于
`data/intent/experiments/curated_v2_round3_graph_ablation/intent_optimization_summary.json`。
已接受的8条难例已经部署到运行时 `data/intent/hard_negatives.jsonl`，实际 Prompt
按配额选取 Additional 3条与 GraphRAG 3条。

## 二阶段 B/C 边界裁决

固定 few-shot 在第三轮已经进入平台期，因此在线路由改为两阶段：第一阶段仍由 DeepSeek
输出五类概率；当 `additional-query + graphrag-query` 概率不少于0.45时，第二阶段专门
解析当前问题和对话上下文中的商品/订单对象，并判断下一步应当追问还是直接检索。原始
B/C 概率与裁决结果各占0.5，避免裁决器过度偏向 GraphRAG。

| 数据集 | Accuracy | Macro-F1 | General F1 | Additional F1 | GraphRAG F1 | 错误数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation（单阶段第三轮） | 0.7747 | 0.7818 | 0.9000 | 0.6307 | 0.8148 | 0 |
| validation（二阶段0.5融合） | **0.8671** | **0.8492** | 0.8966 | **0.7431** | **0.9079** | 0 |
| locked test（二阶段，仅评测一次） | **0.8225** | **0.8043** | 0.8659 | 0.6807 | 0.8663 | 0 |

validation上的Macro-F1提升0.0674，锁定test仍保持0.8043，说明提升可以泛化，但不得
把validation的0.8492当作test成绩。二阶段会增加一次API调用：本次validation中526/617
条触发边界裁决。完整报告位于：

- `data/intent/experiments/curated_v2_boundary_blend05/intent_optimization_summary.json`
- `data/intent/experiments/curated_v2_boundary_blend05_test/intent_optimization_summary.json`

本地字符TF-IDF与DeepSeek概率融合也做过独立验证，但validation Macro-F1只有0.7568，
因此未部署。线上配置使用 `INTENT_ROUTER_MODE=boundary_v2` 和
`INTENT_BOUNDARY_BLEND_WEIGHT=0.5`。
