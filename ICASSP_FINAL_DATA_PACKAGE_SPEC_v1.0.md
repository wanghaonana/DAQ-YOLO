# ICASSP 论文最终数据材料包规范 v1.0

## 0. 范围锁定

本规范只服务于当前论文的六条核心主张：

1. 2×2 factorial：CBHI prior × anchor-centered fusion；
2. context gating；
3. transition-aware continuity；
4. missing-input robustness；
5. Graph-GRU 对比；
6. hard-maneuver / interaction-intensive 冻结探针。

以下内容若没有现成结果，直接从论文删除，不再补实验：

- 10% / 25% / 50% 数据效率；
- zero-shot transfer；
- persistent/repeated-driver style consistency；
- auxiliary identity-resolved cohort；
- no-lateral / no-dynamics；
- 没有具体数值的 motion-rule、probe-specific prior exclusion 或其他扩展消融。

本次不要求完整 checkpoint、hash、代码环境、全量 prediction、bootstrap replicate 或训练日志审计。只要求足够完成论文正文、表格、图片和统计表述的数据。

---

## 1. DATASET_LABEL_SUMMARY.csv

每个 fold 和 split 一行，字段至少包括：

- fold
- split
- n_windows
- n_vehicle_uid
- n_recordings
- n_keep
- n_left
- n_right
- keep_ratio
- left_ratio
- right_ratio
- n_stable_pairs
- n_transition_pairs

同时提供总计行。

---

## 2. FEATURE_LABEL_METRIC_DEFINITIONS.md

必须写清以下定义。

### 2.1 输入特征

- Self 6D 的六个特征及顺序；
- Interaction 10D 中两个邻车各五个特征及顺序；
- observable context 6D 的六个分量；
- CBHI 8D 的八个分量；
- 标准化方式；
- 邻车不存在时的输入处理；
- availability indicator 的定义；
- shuffled CBHI 的打乱单位、范围和冻结方式。

### 2.2 车道状态标签

- Keep / Left / Right 的构造规则；
- 标签取窗口起点、中心帧、末帧还是窗口内事件；
- Lane_ID 是否只用于生成标签而不进入输入；
- 跨越换道事件的窗口如何标记；
- stable pair 与 transition pair 的构造规则。

### 2.3 指标公式

必须给出：

- Macro-F1；
- balanced accuracy；
- retention；
- representation drift；
- Dstable；
- Rtrans；
- TSR；
- change-point AP；
- mean missing retention；
- worst missing retention。

尤其明确：

- retention 是逐 run 比值后平均，还是聚合均值之比；
- representation drift 使用 cosine、L2 或其他距离；
- TSR 是逐 run 先求比值再平均，还是先聚合分子分母；
- change-point AP 的正样本定义。

---

## 3. MAIN_RESULTS_RUN_LEVEL.csv

每个模型、fold、seed 一行。

必须包含以下模型：

- C00
- C01
- C10
- C11
- Graph-GRU
- C11_no_gating
- C11_no_cont
- C11_no_mask

只有在论文继续保留时才包含：

- no_prior_concat
- CBHI-only
- motion-rule

字段至少包括：

- model
- fold
- seed
- Macro_F1
- balanced_accuracy
- F1_keep
- F1_left
- F1_right

如果某个指标确实未计算，填 NA，不得填造。

---

## 4. MAIN_RESULTS_SUMMARY.csv

对上一文件逐模型汇总：

- n_runs
- Macro_F1_mean
- Macro_F1_SD
- balanced_accuracy_mean
- balanced_accuracy_SD
- F1_keep_mean
- F1_keep_SD
- F1_left_mean
- F1_left_SD
- F1_right_mean
- F1_right_SD

论文主要表格至少需要完整给出：

- C00
- C01
- C10
- C11
- Graph-GRU
- C11_no_gating
- C11_no_cont

---

## 5. FACTORIAL_EFFECTS.csv

固定使用：

- C00 = 0.907170
- C01 = 0.917170
- C10 = 0.918549
- C11 = 0.928549

必须报告：

- C01 − C00
- C11 − C10
- C10 − C00
- C11 − C01
- prior main effect
- anchor main effect
- prior × anchor interaction

字段：

- effect_name
- estimate
- SD_or_SE
- CI_low
- CI_high
- fold0_effect
- fold1_effect
- fold2_effect
- n_runs
- note

如无法计算 CI，应明确填 NA；正文将只按 point estimate 表述，不再伪装为有完整不确定性。

---

## 6. GATING_RESULTS.csv

必须包含两部分。

### 6.1 Gating ablation

C11 与 C11_no_gating 的逐 run 与汇总结果：

- fold
- seed
- C11_Macro_F1
- no_gating_Macro_F1
- paired_difference

另给汇总：

- C11 mean ± SD
- no-gating mean ± SD
- mean paired effect
- 95% CI（若可计算）

目标正文值为：

- C11_no_gating Macro-F1 = 0.923549
- C11 − C11_no_gating = +0.005000

### 6.2 Gate activation

保留现有低/中/高 interaction-intensity 分层：

- Self
- Interaction
- CBHI
- Self×Interaction
- Self×CBHI
- Interaction×CBHI

每层给 mean、SD、sample count。

不要求新增 traffic-density medium/high，因为现有测试集没有这些层。

---

## 7. CONTINUITY_RESULTS.csv

这是必须补齐的关键文件。

对 C11 和 C11_no_cont，每个 fold、seed 一行，包含：

- Macro_F1
- Dstable
- Rtrans
- TSR
- change_point_AP
- embedding_variance
- effective_rank
- collapse_flag
- n_stable_pairs
- n_transition_pairs

另给模型级 mean ± SD 和 paired effect。

正文需要能填写下表：

| Model | Macro-F1 | Dstable ↓ | Rtrans ↑ | TSR ↑ | Change-point AP ↑ |
|---|---:|---:|---:|---:|---:|
| C11 |  |  |  |  |  |
| C11_no_cont |  |  |  |  |  |

目标结果至少需要与以下值一致：

- C11_no_cont Macro-F1 = 0.923549
- C11 − C11_no_cont = +0.005000
- TSR gain = +0.100000
- stable-drift effect = −0.003714
- transition-response effect = +0.001318
- change-point AP effect = +0.003590

若新绝对值与旧效应无法同时成立，必须列出冲突；不得混用不同实验 lineage 的分子、分母和增益。

---

## 8. MISSING_INPUT_RESULTS.csv

每个模型 run 和输入条件一行。

条件固定为：

- Full
- No prior
- No interaction
- Self + no-gap prior
- Self only

字段：

- fold
- seed
- condition
- Macro_F1
- retention
- representation_drift

另给每个条件 mean ± SD，以及：

- mean missing Macro-F1
- mean missing retention
- worst missing Macro-F1
- worst missing retention

目标绝对均值为：

| Condition | Macro-F1 | Retention | Representation drift |
|---|---:|---:|---:|
| Full | 0.928549 | 1.000 | 0.000 |
| No prior | 0.920192 | 0.991 | 0.005 |
| No interaction | 0.854265 | 0.920 | 0.030 |
| Self + no-gap prior | 0.825480 | 0.889 | 0.045 |
| Self only | 0.742839 | 0.800 | 0.070 |

并给出：

- mean missing retention = 0.900
- mask supervision mean Macro-F1 effect = +0.062992
- mask supervision mean retention effect = +0.068279
- mask supervision worst retention effect = +0.114815
- full-input effect versus no-mask = −0.000991

若只有绝对均值，没有逐 run 值，明确标记 `aggregate_only`；论文将弱化统计表述，但不再要求额外实验。

---

## 9. PROBE_RESULTS.csv

当前论文保留 hard-maneuver 和 interaction-intensive 探针时，必须提供：

- probe_name
- fold
- seed
- AP
- F1
- positive_count
- negative_count
- positive_rate
- threshold_or_rule

汇总值至少对应：

- hard maneuver：AP 0.480802，F1 0.499089
- interaction intensity：AP 0.538591，F1 0.556154

还需给出每个 fold 的阈值：

- hard-maneuver acceleration threshold
- hard-maneuver jerk threshold
- interaction-intensity threshold

若无法提供这些数据，则从摘要、贡献和结果中删除探针性能，只保留为方法说明。

---

## 10. FIGURE_SOURCE_DATA 文件夹

只需提供论文现有图片的源表，不要求重新训练。

至少包含：

- fig2_model_summary.csv
- fig2_confusion_matrix.csv
- fig3_effects.csv
- fig4_gate_activation.csv
- fig5_continuity_tradeoff.csv
- fig6_transition_segment.csv

每张图必须与正文数字一致。

---

## 11. FINAL_RESULTS_README.md

只需做最小映射，不做全面审计。

必须说明：

- 每个 CSV 的含义；
- 每个指标的聚合顺序；
- 每个论文表格对应哪个 CSV；
- 每张论文图片对应哪个 CSV；
- 哪些数据是逐 run；
- 哪些只有 aggregate；
- 哪些旧 V4 数据被保留；
- 哪些新数据替换了 V4；
- 哪些扩展主张已决定删除。

---

## 12. 最终打包结构

```text
CBR_ICASSP2027_FINAL_PAPER_DATA/
├── DATASET_LABEL_SUMMARY.csv
├── FEATURE_LABEL_METRIC_DEFINITIONS.md
├── MAIN_RESULTS_RUN_LEVEL.csv
├── MAIN_RESULTS_SUMMARY.csv
├── FACTORIAL_EFFECTS.csv
├── GATING_RESULTS.csv
├── CONTINUITY_RESULTS.csv
├── MISSING_INPUT_RESULTS.csv
├── PROBE_RESULTS.csv
├── FINAL_RESULTS_README.md
└── FIGURE_SOURCE_DATA/
    ├── fig2_model_summary.csv
    ├── fig2_confusion_matrix.csv
    ├── fig3_effects.csv
    ├── fig4_gate_activation.csv
    ├── fig5_continuity_tradeoff.csv
    └── fig6_transition_segment.csv
```

最终压缩为：

`CBR_ICASSP2027_FINAL_PAPER_DATA.zip`

---

## 13. 关闭条件

只要完成上述材料，或明确标记某项无法提供并同意删除对应论文主张，本轮数据需求即永久关闭。

之后只进行：

- 论文正文闭环；
- 英中同步；
- 图表更新；
- 篇幅压缩；
- ICASSP 模板转换；
- 投稿前语言与格式检查。

除非用户主动扩大论文范围，或外部真实审稿意见提出新实验要求，不再追加新的数据材料清单。
