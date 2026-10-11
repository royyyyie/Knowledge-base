# RAG 评测

本目录保存版本化、可离线复现的 RAG 评测资产。

真实 PDF、Word、Excel、HTML 语料的来源/使用权、哈希、gold Case 与正式索引流程见
[跨格式真实语料评测 Runbook](../../docs/runbooks/cross-format-rag-evaluation.md)。
`scripts/audit_rag_eval_corpus.py` 只审计资料准备情况，不生成解析或答案质量分数。

[200 条扩展基准集的历史实测说明](../../docs/rag-benchmark-v2-assessment-2026-09-14.md)记录旧索引结果；[2026-09-16 当前索引检索基线](../../docs/rag-benchmark-v2-baseline-2026-09-16.md)记录 v4 重新入库后的 200 条检索专项。模型服务 HTTP 402 仍阻断完整答案 Judge 测评。
[2026-09-17 BGE-M3 Sparse / ColBERT 隔离实测](../../docs/rag-bge-m3-modes-assessment-2026-09-17.md)
比较同一 BGE-M3 编码器下的 BM25 与模型 Sparse，并记录本机 ColBERT 预检时延；
该结果不替代答案 Judge 或当前默认配置的发布基线。

## 目录

- `rag_smoke_v1.json`：受控 Orion 语料的人工标注 smoke Dataset。
- `corpus/`：可上传到专用测试知识库的 HTML 语料。
- `rag_benchmark_v2.json`：在原有 20 条基础上扩展的 200 条受控合成 Case；新增标签由固定站点事实生成，尚需业务专家盲审，不能称为人工标注生产样本。
- `corpus_benchmark_v2/`：30 份受控 HTML（原有 10 份加 20 份站点档案），当前解析器生成 90 个三级 chunk，用于更有区分度的 Top-10 检索测试。
- `../../scripts/generate_rag_benchmark_v2.py`：确定性生成扩展 Dataset 和语料；`--check` 校验资产未漂移。
- `offline_smoke_observations_v1.json`：用于验证评分 Module 的脱敏 Observation。
- `gates_v1.json`：质量回归门禁。
- `baseline_v1.json`：由上述 Dataset 与 Observation 生成的 offline smoke baseline。
- `judged_live_gates_v1.json`：答案 Judge 完整覆盖与质量门禁，供 CLI `--judge-answers` 或持久 Job 使用。
- `schema/`：Dataset、Observation、GatePolicy 与 Report 的 JSON Schema。

offline smoke baseline 只证明数据契约、指标和回归门禁可以稳定复现，不代表生产知识库质量。真实评测应把 `corpus/` 上传到隔离的测试索引，使用 live Adapter 生成新的 Observation；不要手工修改 Observation 来美化指标。

## 常用命令

```bash
uv run python scripts/evaluate_rag.py validate \
  --dataset evals/rag/rag_smoke_v1.json

uv run python scripts/generate_rag_benchmark_v2.py --check
uv run python scripts/evaluate_rag.py validate \
  --dataset evals/rag/rag_benchmark_v2.json

uv run python scripts/generate_rag_eval_schemas.py --check

uv run python scripts/evaluate_rag.py score \
  --dataset evals/rag/rag_smoke_v1.json \
  --observations evals/rag/offline_smoke_observations_v1.json \
  --gates evals/rag/gates_v1.json \
  --baseline evals/rag/baseline_v1.json \
  --report .artifacts/rag-eval/report.json \
  --markdown .artifacts/rag-eval/report.md \
  --fail-on-regression

uv run python scripts/evaluate_rag.py run \
  --dataset evals/rag/rag_smoke_v1.json \
  --gates evals/rag/live_gates_v1.json \
  --observations .artifacts/rag-eval/live-observations.json \
  --report .artifacts/rag-eval/live-report.json \
  --markdown .artifacts/rag-eval/live-report.md \
  --profile-id local-orion-eval-v1 \
  --index-id your-index-manifest-or-collection-version \
  --timeout-seconds 60 \
  --fail-on-regression

uv run python scripts/evaluate_rag.py run \
  --dataset evals/rag/rag_smoke_v1.json \
  --gates evals/rag/judged_live_gates_v1.json \
  --observations .artifacts/rag-eval/judged-observations.json \
  --report .artifacts/rag-eval/judged-report.json \
  --profile-id local-orion-eval-v1 \
  --index-id your-index-manifest-or-collection-version \
  --timeout-seconds 60 \
  --judge-answers \
  --fail-on-regression
```

`run` 默认只观察生产 RAG/HITL 的检索结果；`--judge-answers` 复用持久 Evaluation Job 的受控答案生成与 Evaluator，不模拟完整 Agent Run 的最终回答。不同评测模式或 Gate policy 的报告与 baseline 不可混用。一次进程只运行一个 profile；修改模型、索引或 RAG 参数后请启动新进程。`profile-id` 与 `index-id` 必须显式填写；报告自动绑定 corpus 字节、RAG/评测源码、Gate policy、评测模式以及脱敏后的模型/Embedding/Rerank/检索配置 fingerprint。源码指纹不包含依赖锁文件；依赖升级改变行为时需提升指纹版本并重建 baseline。报告和 Observation 不保存 chunk 正文、endpoint、密钥或原始异常。

当前 Document build profile 为 `document-loader-v4` / `three-level-800-100-v4`，其中 PDF、Word、Excel 增加格式专用解析；HTML 标题路径和表格行行为维持 v3 的确定性输出。原 200 条测评报告使用更早解析版本和 110 个 chunk，仅作历史记录。2026-09-16 已在独立 Tenant 用当前 profile 重新发布 30 个 Document Version，核对 90 个三级 chunk 和新 `index_id`，并运行 200 条无 LLM 检索专项；详见[当前基线记录](../../docs/rag-benchmark-v2-baseline-2026-09-16.md)。这 200 条 HTML 题目不能用作 PDF、Word、Excel 解析质量的证据。

`judge_coverage_rate` 的分母是已产生 Observation 的 Case；完整报告要求所有 Case 都有 Observation。Markdown 报告同时显示 Judge 完成数。答案质量门禁目前要求 Judge 覆盖率为 1、平均正确性与有据性至少 0.7、无证据声明比例至多 0.2；这些初始阈值必须经人工抽检校准。Provider 失败会保留阶段与稳定错误码，不会被当作正确拒答。`--allow-source-mismatch` 只允许调查源码差异，其 Report 明确失败，不能作为发布门禁。

`gates_v1.json` 只接受 `contract_smoke` provenance；静态 Observation 不能冒充实时质量结果。`live_gates_v1.json` 和 `judged_live_gates_v1.json` 只接受 `live_rag` provenance，后者还要求完整答案 Judge 指标。GitHub 的默认 workflow 验证评分器、baseline 可重建性和基准分支差异；没有配置模型、Milvus 与隔离测试索引的托管 runner 时，它不会宣称已经执行生产 RAG。RAG 发布前仍必须在受控索引上运行 live 命令并审查对应报告。

## 扩充标注集

`rag_benchmark_v2.json` 已有 200 条、30 份文档，适合发现 Top-10 检索在近似站点和跨文档问题上的失效；固定模板带来相似问题的相关性，不能把 200 条视为 200 个独立业务情境。它保留原有时间/版本、来源冲突、代码和文档提示注入 Case，新增单事实、跨节、跨文档、多跳、近似实体、无知识和 HITL 切片。最终生产门禁仍需由业务专家独立抽样、盲审并扩充真实长文、表格、扫描件和多版本 Document Version。当前 `prompt_injection` case 带有 `retrieval_only` 标签，只验证安全说明能被正确检索；在结构化答案与泄密探针接入前，不得据此宣称完整 Guardrail 已通过。

从仓库根目录运行 200 条真实检索测评时，先把 `corpus_benchmark_v2/` 经正式上传、Index Job 和 Document Version 发布到隔离 Tenant/索引，确认所有 30 份文档为 current，再执行：

```bash
uv run --no-sync python scripts/evaluate_rag.py run \
  --dataset evals/rag/rag_benchmark_v2.json \
  --corpus-path evals/rag/corpus_benchmark_v2 \
  --gates evals/rag/live_gates_v1.json \
  --observations .artifacts/rag-eval/benchmark-v2-observations.json \
  --report .artifacts/rag-eval/benchmark-v2-report.json \
  --markdown .artifacts/rag-eval/benchmark-v2-report.md \
  --profile-id your-frozen-profile-id \
  --index-id your-current-index-fingerprint \
  --timeout-seconds 180 \
  --fail-on-regression
```

不得拿原 20 条的 offline baseline 与新 Dataset 作回归比较；两者的 Dataset、corpus、index 和 profile 身份不同。

新增或修改 Dataset 后 fingerprint 会变化，旧 Observation 与 baseline 会被拒绝。真实 corpus/index identity 应在 PR-15 后切换到 DocumentVersion 与 IndexManifest hash。
