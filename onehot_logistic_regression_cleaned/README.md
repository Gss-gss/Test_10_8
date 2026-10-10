# 清洗后 One-hot 逻辑回归基线

本目录汇集正式的 One-hot 逻辑回归基线。训练数据仍从工作区根目录的 `trainPos.txt` 与 `trainNeg.txt` 读取，以便与其他实验共享同一份原始数据。

- 输入：31 位序列，每位 21 维 One-hot，展开为 651 维；
- 清洗：保留全部 3,573 条正样本，删除 2 条非 31 位负样本，保留 3,571 条负样本；
- 分类器：`LogisticRegression(max_iter=3000, solver="liblinear")`；
- 评估：固定随机种子 20261009 的分层五折交叉验证；
- 结果：SN 0.5617 ± 0.0153，ACC 0.5601 ± 0.0109，SP 0.5584 ± 0.0198，MCC 0.1201 ± 0.0219。

从工作区根目录运行：

```powershell
python .\onehot_logistic_regression_cleaned\run_onehot_logistic_regression_cv.py
```

五折结果保存在 `results/onehot_logistic_regression_5fold.csv`。未清洗的历史结果 `onehot_5fold_results.csv` 保留在根目录，不属于正式基线。
