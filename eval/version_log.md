# 评测版本日志


## 2026-10-30 22:21:45 - baseline (qwen3-8b, v1.1)

- **准确率**: 70.6%
- **总问题数**: 68
- **Bad case数量**: 20
- **评测集版本**: v1.1
- **模型**: qwen3-8b
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-8b/v1.1/baseline_qwen3-8b_v1.1_20261030.json`

## 2026-11-01 12:25:24 - baseline (qwen3-14b, v1.1)

- **准确率**: 73.5%
- **总问题数**: 68
- **Bad case数量**: 18
- **评测集版本**: v1.1
- **模型**: qwen3-14b
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b/v1.1/baseline_qwen3-14b_v1.1_20261101.json`

## 2026-11-01 13:16:23 - baseline (qwen3-14b, v1.1)

- **准确率**: 75.0%
- **总问题数**: 68
- **Bad case数量**: 17
- **评测集版本**: v1.1
- **模型**: qwen3-14b
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b/v1.1/baseline_qwen3-14b_v1.1_20261101.json`

## 2026-11-01 14:05:13 - baseline (qwen3-14b, v1.1)

- **准确率**: 83.8%
- **总问题数**: 68
- **Bad case数量**: 11
- **评测集版本**: v1.1
- **模型**: qwen3-14b
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b/v1.1/baseline_qwen3-14b_v1.1_20261101.json`

## 2026-11-01 20:05:26 - sft (qwen3-14b-sft, v1.1)

- **准确率**: 75.0%
- **总问题数**: 68
- **Bad case数量**: 17
- **评测集版本**: v1.1
- **模型**: qwen3-14b-sft
- **阶段**: sft
- **结果文件**: `results/sft/qwen3-14b-sft/v1.1/sft_qwen3-14b-sft_v1.1_20261101.json`

## 2026-11-01 21:26:23 - baseline (qwen3-14b-nothink, v1.1)

- **准确率**: 80.9%
- **总问题数**: 68
- **Bad case数量**: 13
- **评测集版本**: v1.1
- **模型**: qwen3-14b-nothink
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b-nothink/v1.1/baseline_qwen3-14b-nothink_v1.1_20261101.json`

## 2026-11-04 15:52:50 - baseline (qwen3-14b-think, v1.2)

- **准确率**: 83.8%
- **总问题数**: 68
- **Bad case数量**: 11
- **评测集版本**: v1.2
- **模型**: qwen3-14b-think
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b-think/v1.2/baseline_qwen3-14b-think_v1.2_20261104.json`

## 2026-11-04 16:02:14 - baseline (qwen3-14b-nothink, v1.2)

- **准确率**: 83.8%
- **总问题数**: 68
- **Bad case数量**: 11
- **评测集版本**: v1.2
- **模型**: qwen3-14b-nothink
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b-nothink/v1.2/baseline_qwen3-14b-nothink_v1.2_20261104.json`

## 2026-11-05 11:36:39 - baseline (qwen3-14b-think, v1.2)

- **准确率**: 85.3%
- **总问题数**: 68
- **Bad case数量**: 10
- **评测集版本**: v1.2
- **模型**: qwen3-14b-think
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b-think/v1.2/baseline_qwen3-14b-think_v1.2_20261105.json`

## 2026-11-05 11:48:59 - baseline (qwen3-14b-nothink, v1.2)

- **准确率**: 83.8%
- **总问题数**: 68
- **Bad case数量**: 11
- **评测集版本**: v1.2
- **模型**: qwen3-14b-nothink
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b-nothink/v1.2/baseline_qwen3-14b-nothink_v1.2_20261105.json`

## 2026-11-06 - C1 审核统一修复批次（规则 + 工具 + mock + GT，待 AutoDL 重跑基线）

C1 三批人工审核（hallucinated_args / false_promise+missing_evidence）全部完成后的一揽子修复：

**规则层（eval/verifier/rules.py）**
- 问题 7：observation 落盘为 JSON 字符串致 startswith 空结果判断失效 → 新增 `_obs_has_data`（JSON 感知：data.total/results + text 空结果表述子串 + 纯文本错误前缀）替换两处判据；`_echoes_failure` 补 success 但空结果的豁免（text_search 查无结果 status=success）
- 问题 13（A3 回放新发现）：claim 词否定形式误匹配 → 新增 `_negated_at` 否定豁免（"暂未查询到"不误报 no_side_effect、"没有预约成功"不误报幻觉预约）
- ④ 方案乙：TRACE_ARGS 白名单 car_model 入、car_price 移（溯源对象从车价变车型）

**判分脚本（eval/baseline_eval.py）**
- 问题 12：`_trace_args_source` 纯子串匹配 → 新增 `_value_in_text`（数字 token 精确 + 负向后行排车型名 + 短 ASCII 词词边界 + 长串子串），"0"不再命中"90天"
- 落盘修复：tool_response_raw 完整落盘（去掉 2048 字符截断）

**工具与数据**
- 问题 10：`tools/calculate_finance.py` 重写为自动读价（car_model 必填、显式 car_price 拒绝、score_match 解析指导价：1 条取价/多条 partial 列配置/0 条 NOT_FOUND、反查优惠"另有优惠 Y 未计入"）
- 问题 9：`tools/mock_data/query_discount_policy.json` 帕萨特/小米补"全国"键
- 问题 11：`questions.json`（生效版）acceptable_sets 加等价路径（compare_004/008 加 ["text_search"]、multi_hop_002 加 ["text_search","query_discount_policy"]）；`questions_v1.2.json` 为冻结快照不回写

**验证（A3 校准回放）**
- 单元测试 29/29 全绿（含新增 T10-T15：问题 7 三向 + 否定豁免三向）
- calculate_finance 8 场景单测通过；discount_004/009 实测从"未找到"变"查到全国优惠"
- 离线重判（`_a3_rejudge_report.json`）：负例 5/5 干净；正例 8/20 命中（修复前即 10/20，EXPECTED 标签过时；本次变化 3 条方向全对：car_config_004 清零✓、car_config_001 去 no_side_effect✓、compare_008 因 GT 等价路径由 missing_evidence 变干净✓）
- 待 AutoDL 重跑双基线验证全量效果

## 2026-11-06 11:36:16 - baseline (qwen3-14b-think, v1.2)

- **准确率**: 85.3%
- **总问题数**: 68
- **Bad case数量**: 10
- **评测集版本**: v1.2
- **模型**: qwen3-14b-think
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b-think/v1.2/baseline_qwen3-14b-think_v1.2_20261106.json`

## 2026-11-06 11:43:19 - baseline (qwen3-14b-nothink, v1.2)

- **准确率**: 86.8%
- **总问题数**: 68
- **Bad case数量**: 9
- **评测集版本**: v1.2
- **模型**: qwen3-14b-nothink
- **阶段**: baseline
- **结果文件**: `results/baseline/qwen3-14b-nothink/v1.2/baseline_qwen3-14b-nothink_v1.2_20261106.json`
