# One-hot 编码分类器对比（不重采样）

本实验将长度为 31 的氨基酸窗口编码为 31 × 21 = 651 维 One-hot 向量，并比较逻辑回归、LightGBM、RBF-SVM 与 GPU MLP。

- 样本预处理：仅保留长度为 31 且中心第 16 位为 K 的窗口。
- 重采样：关闭；过滤后正、负样本分别为 3,573、3,571。
- 验证：嵌套分层交叉验证，外层五折用于最终报告，内层三折选择每个模型的参数；外层测试折不参与标准化、调参或模型选择。
- 评分：内层模型选择依据 AUC；最终报告 SN、ACC、SP、MCC、AUC。
- SVM：用 RBF 核决策分数计算 AUC，并通过 sigmoid 映射到 0.5 分类阈值，不做额外概率校准交叉验证。

运行：

```powershell
.\onehot_bilstm_attention\.venv_gpu\Scripts\python.exe .\onehot_classifier_comparison\run_onehot_classifier_comparison.py
```

结果会写入 `results/`。
