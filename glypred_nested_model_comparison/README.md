# 融合特征模型对比（不重采样）

本目录在同一五种融合特征上公平比较三类模型：LightGBM、RBF-SVM 和 MLP。

- 数据：只保留 31 位且中心为 K 的窗口；两条异常负序列被过滤。
- 重采样：关闭。筛选后的正、负样本分别为 3,573 和 3,571，已近似平衡。
- 特征：AAC、PWAA、EBGW、二肽 KMER、DR，合计 1,841 维。
- 特征选择：每个内层训练分区和最终外层训练折独立使用 LightGBM `split` 重要性，保留大于 2 的特征。
- 评估：外层分层五折用于最终测评；每个外层训练折内再进行分层三折，按验证 AUC 选择模型和参数。外层测试折从不参与特征选择、标准化或调参。
- SVM：使用 RBF 核的决策分数计算 AUC，并将其经 sigmoid 映射后以 0.5（即原始决策分数 0）作为分类阈值；不做额外的概率校准交叉验证。

运行：

```powershell
.\onehot_bilstm_attention\.venv_gpu\Scripts\python.exe .\glypred_nested_model_comparison\run_nested_model_comparison.py
```

结果将写入 `results/`。
