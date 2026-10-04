# 正文图表的数据位置

数据路径相对 `data/`；当前全部图的唯一绘图实现是 `../reproduction/code/publication_figures.py`。图稿导出PDF/SVG/600 dpi PNG，来源、单位、不确定性及几何检查记录在 `../reproduction/output/publication_figure_contracts.json`。旧版figure_data表仍留作原稿追溯。

| 图 | PDF | 科学来源 |
|---|---|---|
| 1 | `principle.pdf` | 方法示意；`publication_figures.py:principle`，无测量值 |
| 2 | `ambiguity_example.pdf` | `publication_figures.py:ambiguity`中的解析双世界构造；独立验证见`results/identifiability/two_world_verification.json` |
| 3 | `audit_evidence.pdf` | `results/identifiability/all_model_audit_summary.csv`；`results/benchmark/audit_cases.csv`、`ambiguity_impact.csv` |
| 4 | `complete_global.pdf` | `results/global_and_prediction/` 的 exact/block/verified/global/coupled 摘要与证书 |
| 5 | `prediction_safe.pdf` | `results/global_and_prediction/final_safe_summary.csv` |
| 6 | `sensitivity.pdf` | `results/sensitivity/radius_summary.csv`、`calibration_size_summary.csv` |
| 7 | `multiclass_decisions.pdf` | `results/multiclass_and_calibration/multiclass_verified_cases.csv`、`multiclass_verified_candidates.csv`，严格核验结果 |
| E.1 | `calibration_evidence.pdf` | `results/policy_controls/oracle_policy_test.csv`、`learned_oracle_calibration.csv`；已知机制控制移至附录E |

| 表 | 科学来源 |
|---|---|
| 1：任务 | `run_experiments.py` 和 `gap_experiments.py` 的数据定义 |
| 2：动作证书 | `results/global_and_prediction/global_minimax_summary.csv`、`global_all_action_bounds.csv` |
| 3：预测比较 | `results/global_and_prediction/final_safe_summary.csv` |
| 4：运行成本 | `results/sensitivity/local_runtime_summary.csv`、`local_runtime_batches.csv` |
| 附录表D.1：静态规则 | `results/global_and_prediction/final_static_diagnostic_summary.csv` |

图3的误差/分布单位为配置而非候选行；图4的区间是数学目标包络；图5的误差棒为独立测试Monte Carlo SE，校准下界另列；图6前缀样本嵌套；图7为30个设置的完整核验；图E.1涉及已知机制的系统级覆盖与描述性差值。具体统计定义随正文图注保留。
