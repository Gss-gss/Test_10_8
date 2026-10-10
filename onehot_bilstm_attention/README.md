# One-hot 加 Glypred 风格 BiLSTM 注意力分类器

本实验保持 31 位序列的 21 通道 One-hot 编码，替换原逻辑回归分类器为论文描述的改进 BiLSTM 分类器：双向 LSTM、LayerNorm、多头注意力、GELU、Dropout 0.5 与全连接残差连接。

## 训练协议

- 长度不为 31 或中心第 16 位不为 K 的记录会被剔除。
- 使用随机种子 20261009 的分层五折交叉验证。
- 每个训练折额外划分 10% 为验证集，用验证损失早停；测试折不参与早停或模型选择。
- Adam：学习率 0.01，L2 权重衰减 0.005，batch size 64。
- 最多训练 50 个 epoch，验证损失连续 8 个 epoch 不改善时停止并恢复最佳权重。为使小批量 LSTM 在 CPU 上稳定运行，训练限制为最多 4 个 CPU 线程。

## 运行

```powershell
.\onehot_bilstm_attention\.venv_gpu\Scripts\python.exe .\onehot_bilstm_attention\run_onehot_bilstm_attention_cv.py
```

本目录的 `.venv_gpu` 已安装 PyTorch 2.11.0 + CUDA 12.8，并可使用本机的 NVIDIA GPU。脚本会自动选择 CUDA；如果该环境被替换为 CPU 版 PyTorch，脚本会自动回退到 CPU。

输出文件位于 `results/`：

- `run_audit.json`：数据筛选、编码、网络和训练配置。
- `onehot_bilstm_attention_5fold.csv`：每折混淆矩阵、SN、ACC、SP、MCC、AUC 与早停信息。

该实现复现论文所描述的分类器组件；论文没有公开完整的层宽、早停耐心值和验证集切分比例，因此它们在脚本中明确设定为可复现配置，而非声称与作者训练权重完全一致。
