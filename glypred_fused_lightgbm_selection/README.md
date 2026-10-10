# 五特征融合加 LightGBM 特征选择基线

本目录是独立实验：在 31 位、中心 K 的序列窗口上提取 AAC、PWAA、EBGW、KMER(k=2) 和 DR(d=3) 五类特征，再在每一折训练集内使用 LightGBM 的 `split` 重要性选择特征，最后以逻辑回归分类。

## 关键约束

- **不使用 CCU**：当前数据集正负样本近似平衡，CCU 会无谓地删除训练信息。
- **没有特征选择泄漏**：LightGBM 只在每一折的训练数据拟合；测试折从未参与特征重要性计算。
- **固定阈值**：保留 `split` 重要性大于 2 的特征，与论文的阈值 2 设置对齐。若要调这个阈值，应使用嵌套交叉验证。

## 运行

```powershell
.\.venv\Scripts\python.exe .\run_fused_lightgbm_selection_logreg_cv.py
```

输出保存在 `results/`：

- `feature_audit.json`：样本、特征和模型配置。
- `fused_lightgbm_selected_logreg_5fold.csv`：逐折指标与均值、标准差。
- `selected_feature_families_by_fold.csv`：每折及每类特征保留数量。
