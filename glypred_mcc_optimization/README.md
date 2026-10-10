# 融合特征 LightGBM 的 MCC 优化实验

本目录沿用 31 位序列与五种融合特征，比较 MCC 定向优化方案和上次最佳 LightGBM 的固定 0.5 阈值方案。正、负样本近似均衡，因此不重采样。

优化方案在每个外层训练折内执行三折验证：每次重新筛选特征，再训练六组 LightGBM 参数并取得训练折的完整折外预测。根据这些折外预测选择参数和分类阈值，最后仅在相应外层测试折评估一次。特征筛选器的 `split` 重要性阈值固定为大于 2。

比较基准与此前最佳方案相同：学习率 0.02、15 个叶子、最小子节点样本数 30、400 棵树、固定概率阈值 0.5。同时单独报告仅优化阈值的消融结果。三个方案使用相同的五个外层测试折。

运行：

```powershell
.\onehot_bilstm_attention\.venv_gpu\Scripts\python.exe .\glypred_mcc_optimization\run_mcc_optimization.py
```

结果输出在 `results/` 中，包含逐折结果、内层候选评分及数据审计。实际结果和解释见 `RESULTS.md`。
