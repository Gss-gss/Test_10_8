# One-hot BiLSTM：AdamW + Warmup + Cosine decay

本目录是对 `onehot_bilstm_attention_tuned` 的独立优化对比实验。模型结构、31 × 21 One-hot 输入、四组候选超参数、内部验证集选择方式、早停规则和外层分层五折均保持一致；唯一改变是训练优化策略：

- 优化器从 Adam 改为 AdamW；
- 前 5 个 epoch 将学习率从候选初始学习率的 10% 线性升至初始学习率；
- 此后学习率按余弦曲线衰减，至第 60 个 epoch 的最小值为 `1e-5`；
- 每个外层训练折都独立执行该调度；外层测试折不参与选择或早停。

## 选择该实验的原因

AdamW、Warmup 和 Cosine decay 适用于通过梯度更新参数的神经网络，因此应加入 One-hot BiLSTM 注意力模型，而不应加入逻辑回归、SVM 或 LightGBM：后两者不是按神经网络小批量梯度训练，Warmup/余弦学习率也不适用。

此前直接迁移论文训练参数时，One-hot BiLSTM 出现类别塌缩；训练折内调参后模型恢复为有效分类器，但 SN/SP 的折间波动依然较大。该策略的作用是让 BiLSTM 在训练初期避免过大的更新破坏随机初始化表示，并在中后期用逐步减小的步长细化参数；AdamW 将权重衰减与自适应梯度更新解耦，通常比 Adam 的 `weight_decay` 更便于控制正则化。

该实验的结果应仅与 `onehot_bilstm_attention_tuned` 的原 Adam 版本直接比较，因为两者的数据、结构和嵌套验证设计相同。是否优于融合特征 LightGBM，仍以外层 MCC、AUC 和折间波动共同判断。

## 运行

```powershell
.\onehot_bilstm_attention\.venv_gpu\Scripts\python.exe .\onehot_bilstm_adamw_warmup_cosine\run_adamw_warmup_cosine_nested_cv.py
```

输出写入本目录的 `results/`：

- `run_audit.json`：数据清洗、设备和调度配置；
- `onehot_adamw_warmup_cosine_5fold.csv`：外层五折结果；
- `inner_candidate_validation.csv`：全部训练折内候选表现。
