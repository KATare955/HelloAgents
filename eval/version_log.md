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

## 2026-11-06 - C2 判定层修复批次（think 块剥离 + 否定豁免扩展 + 预约时间移出白名单）

AutoDL 重跑结果逐题对比后暴露的 4 个规则层缺陷修复（全部离线可重判，无需重跑模型）：

**规则层（eval/verifier/rules.py）**

- 问题 14：think 模式 64/68 题回复含 `<think>` 独白块，规则在独白上错罚（21 题"暴露工具名"全来自独白、multi_hop_004 负例被独白否认污染）→ 新增 `_strip_think_block`，判定入口先剥掉独白块再跑各条规则
- 问题 15："无法查询到"前字为"法"漏豁免 → `_negated_at` 两字前缀补"无法/未能/不能/没法"
- 问题 16：预约时间由模型换算相对时间，字面溯源必失败 → `appointment_time` 移出 TRACE_ARGS（book_test_drive 保留车系/经销商/姓名/电话）
- 问题 17："查询到"挂在 3 个查询工具名下，未调用的 text_search 误认领 → `_cap_no_side_effect` 实际调用工具先认领自己的 claim 词

**验证**

- 单元测试 41/41 全绿（新增 T16-T21）
- A3 校准回放：负例 5/5 干净；正例漏抓 12 题均为"修复后变干净"（方向正确）
- 离线重判（20261106 批）：think verify 满分 27→**49/68**（+22）；nothink 44→**49/68**（+5）
- 残留坏题（think 18 / nothink 19）以真阳性为主：模型编造贷款参数（题面未给首付/年限/利率）、编造预约姓名电话、链路偏离 GT——为 SFT 负面样本来源；car_config_008 关联问题 8 待排查

## 2026-11-06 - 问题 18 修复（默认利率 4.5% 豁免裁定落地）

**规则层（eval/verifier/rules.py）**

- 问题 18：C1 第二批口径 B 已裁定豁免 4.5%（calculate_finance 工具 interest_rate 参数 schema 明示 default=4.5，模型传 4.5 与不传等价、属遵守工具契约），但裁定只记在审核清单未落地到判分规则 → TRACE_ARGS 中 `calculate_finance` 移出 `interest_rate`（首付比例/贷款年限仍受检——工具必填且无默认值，挂决策点 A）

**验证**

- 单元测试 44/44 全绿（新增 T22：interest_rate=4.5 传默认值不判编造；首付/年限编造仍判）
- 离线重判（20261106 批第三次）：think 49→**52/68**（+3）、nothink 49→**52/68**（+3）；multi_hop_001/003/007（双模式 5 题次）+ finance_010（think）共 6 题次恢复满分，0 题下降
- 残留坏题：think 16 / nothink 16（共 32 题次）；问题库逐题定性表已同步（6 题次移出靶子）

## 2026-11-06 - 重跑前统一修复批次（问题 8/19 + 决策点 A/B/C 落地）

**背景**：问题库审核定案后，四项裁定涉及工具/mock/GT 修改，为让今晚重跑一次拿齐"最终干净基线"，本批次把全部重跑前必改项落地（4 个决策点全部执行：A1 工具默认值 / B 反问合格豁免 / C3 mock 补续航 / D 问题 8 修复）。

**工具层（tools/）**

- 问题 8（决策点 D）：text_search 兜底 `s > 0` 全兜底 → 收紧为"实质 token 重叠"（新增 `_matching.has_substantive_overlap`，共享 token 至少一个非纯年份/非中文通用单字）。假车型"星曜X7 2025款"从返回 8 条噪声（实测得分 0.571/0.500/0.444/0.222 全因"2025""款"重叠）变为"未找到+可用车型列表"（total=0）。回归：双基线轨迹 36 个去重真实查询全部行为一致
- 决策点 A（A1）：calculate_finance 首付比例/贷款年限改为可选参数（default 30%/3年），缺省按默认方案计算且返回标注"按默认方案计算（首付30%、3年期），可调整"——把确定性留给工具，模型无需猜或反问
- 决策点 C（C3 mock 侧）：competitor_compare.json 补"续航"维度（E5 旗舰quattro型 623km vs 小米SU7 四驱Max版 800km，与 text_search mock 的 range_cltc 同源），summary 优势计数与 recommendation 同步

**评测层（eval/）**

- 问题 19：评测从未注入日期基准（查证：system prompt/题目/框架模板/工具/mock 均无），模型凭训练记忆把"今天"猜成 2023 年（v1.1 实锤 7 题）→ ① baseline_eval.py system prompt 注入"当前日期：{EVAL_TODAY}（周X）"（环境变量 EVAL_TODAY 可固定，默认运行当天，env_snapshot 记录 eval_today）；② 判分器恢复预约时间**年份正确性**检查（book_test_drive.appointment_time 年份≠基准年份→hallucinated_args，非字面溯源）；③ 同时修正问题 16 排查记录中"日期基准 2023-10-05"的未验证推断（初版结论有误）
- 决策点 A（判分侧）：TRACE_ARGS calculate_finance 移出 down_payment_ratio/loan_years（与问题 18 interest_rate 同口径：工具已有默认值，传默认值/不传属遵守工具契约），car_model 仍受检（唯一事实入口）
- 决策点 B：questions.json test_drive_005 加 accept_clarification（target_tool=book_test_drive，缺参反问合格）；验证：think 反问 1.0 满分、nothink 编造硬调仍罚 0.25（豁免只放行"没调 target 且反问"，nothink 调了 target 不受影响）
- 决策点 C（GT 侧）：compare_005 GT 简化（competitor_compare mock 已可答全三方面，移出 text_search，type 双跳→单跳，gt_note 记录裁定；"降级补查"能力改由 SFT 数据示范）

**验证**

- 单元测试 49/49 全绿（T22 更新：首付/年限默认值不判编造+车型编造仍判；新增 T23 年份正确性检查、T24 假车型兜底收紧）
- 工具自测：calculate_finance 不传首付/年限 → 默认方案标注"可调整"；显式传 50%/5年 → 无默认标注
- 离线重判（旧轨迹 × 全部新规则，= 修前基线）：judge 口径 think 59/68（翻转 1 题：test_drive_005 think 反问合格翻正）、nothink 59/68；verify 满分 think 53/68、nothink 53/68（年份幻觉 3+4 题现形属预期——旧轨迹在无日期注入环境跑的）
- **今晚重跑预期**：日期注入后 test_drive 系列年份恢复 2026（+3/+4）；问题 8 修复后 car_config_008 双模式如实转述不再违规；金融默认值后 M01 类题次工具返回带"可调整"标注、模型照实转述即合规
