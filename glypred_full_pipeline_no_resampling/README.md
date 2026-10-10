# Glypred 论文完整流程改进实验 不重采样

本目录以论文的完整建模路线为基础：

1. 仅保留长度为 31、中心第 16 位为 K 的窗口；
2. 连接 AAC、PWAA、EBGW、二肽 KMER 和 DR 五类特征；
3. 在每个外层训练折中使用 LightGBM 的 `split` 重要性，保留重要性大于 2 的特征；
4. 将选中并按训练折标准化的特征序列输入改进 BiLSTM 分类器；该分类器包含 LayerNorm、多头注意力、GELU、Dropout、残差连接与早停；
5. 用分层五折给出 SN、ACC、SP、MCC 和 AUC。

## 明确不做的步骤

不使用 CCU 或其他重采样。当前筛选后数据包含 3,573 条正样本和 3,571 条负样本，近似均衡；在此条件下欠采样会删除有效训练信息。

论文使用独立测试集，但当前工作区只提供训练数据。因此本实验以分层五折替代独立测试集；所有 LightGBM 选择和早停均只在外层训练折内执行。

## 运行

复用已安装的 GPU 环境：

```powershell
.\onehot_bilstm_attention\.venv_gpu\Scripts\python.exe .\glypred_full_pipeline_no_resampling\run_full_pipeline.py
```

结果会写入 `results/`。
