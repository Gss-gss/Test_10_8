# One-hot 加调参 BiLSTM 注意力分类器

本实验保留 31 × 21 One-hot 输入与论文风格分类器：BiLSTM、LayerNorm、多头注意力、GELU、Dropout 和残差连接。它不再把论文的训练超参数直接套用于 One-hot 输入，而是在**每个外层训练折内**选择超参数。

## 无泄漏评估流程

1. 外层分层五折：当前折测试集完全保留到最后。
2. 每个外层训练折：再分出 15% 内部验证集。
3. 四组学习率、L2 权重衰减、Dropout 和隐藏层大小候选均只在内部训练/验证数据上比较。
4. 以验证 AUC 选择候选和早停 epoch。
5. 用选定的配置和 epoch 在完整外层训练折重新训练；仅一次评估外层测试折。

因此，测试折不会参与早停或超参数选择。

## GPU 运行

复用前一个实验目录内已验证的 CUDA 环境：

```powershell
.\onehot_bilstm_attention\.venv_gpu\Scripts\python.exe .\onehot_bilstm_attention_tuned\run_nested_tuned_cv.py
```

输出在 `results/`：`run_audit.json`、外层五折结果 CSV，以及全部内部候选的验证结果 CSV。
