# 论文数据

| 目录 | 内容 |
|---|---|
| `results/benchmark/` | 600 个预测接口、五步查询、1,246 个可行联合模型及预测数组 |
| `results/identifiability/` | 双世界构造与遗憾下界 |
| `results/policy_controls/` | 匹配策略、机制真值、校准与规模控制 |
| `results/global_and_prediction/` | 全局优化、精确证书、全部最终确认推断与损失 |
| `results/multiclass_and_calibration/` | 32 候选扩展、三分类、信息及联合校准 |
| `results/sensitivity/` | 修复容差、校准样本量和运行成本 |
| `figure_data/` | 解析例子与三分类图的派生源表 |

图表对应位置见 [figure_sources.md](figure_sources.md)。最终损失以 `final_safe_outcomes_compact.npz` 无损保存，结果表保留全部十二个任务—基线比较。

公开任务使用 scikit-learn 的 breast cancer、Iris、wine、digits 加载器；合成任务由代码与种子定义。预测文件包含样本、开发集、候选和目标索引。
