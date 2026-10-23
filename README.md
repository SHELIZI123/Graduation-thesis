# Graduation-thesis · V17.3

面向精神心理健康评估的脑电表征学习与多模态融合方法研究。

本仓库当前发布 **V17.3**：EEG、眼动、心率相关特征作为输入，PSQI、SCL-90、EPQ 的既有评分作为回归目标。V17.3 已替换仓库工作区中的 V10；旧版仍可通过 Git 历史回溯。

## 本版方法

- **强参照与残差学习**：在每个训练子集内分组三折，从 `[1,10,100]` 选择 Ridge 收缩强度；残差交叉拟合也重新选参，留出标签不参与拟合或参数选择。
- **公平的 EEG 表征比较**：统计、黎曼切空间、冻结 CBraMod 表征采用相同 EEG 可用人群、最多32个主成分及相同回归头。
- **多模态与多任务对照**：固定能量的特征块融合，以及普通残差、辅助监督、MMPareto 三个递进候选。零修正是显式已拟合参照，不强迫网络输出修正。
- **评估与可审计推理**：受试者分组的外层五折、完整内层选参、三种子折内集成；保留逐折决策及预测重放检查。TabPFN-v2 使用显式 FP32、math attention 和查询特征快照。

`code/train_v173.py` 是当前训练入口，`code/evaluate_v173.py` 负责合并与复核，`code/refit_v173.py` 负责全开发集重拟合，`code/infer_v173.py` 负责无需量表文件的推理。`*_v171.py` 和 `*_v172.py` 中部分模块是当前版本依赖，不能直接删除。

## 安装和无患者数据测试

Python 3.11。先安装与服务器匹配的 PyTorch；已有 CUDA 12.1 环境可使用：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.foundation.txt
python -m pip check
python -B verify_public_source.py
python -B run_public_tests.py
```

公开测试使用人工生成的数值夹具和代码内置的评分定义，不读取、生成或替代任何患者记录，不需要权重或队列数据。数值测试覆盖分组隔离、目标缺失、训练内变换、残差交叉拟合、回退一致性和特征匹配。它们只证明工程性质，不代表临床效果。

## 导入已有的私有服务器包

GitHub **不包含**患者信号、受试者清单、量表值、特征、逐人预测、训练模型或预训练权重。要复跑既有队列，请先通过受控私有途径将原 V17.3 服务器 ZIP 解压到仓库外，再执行：

```bash
python -B prepare_runtime.py --source /data/private/trimodal_v17_3_20261022
python -B code/check_package_v173.py --foundation
```

导入脚本先校验原包清单、全部原文件及公开源码哈希，再复制 `data/`、`extras/`、`weights/` 和继承快照；不会修改源包。相同文件允许重复导入，已有不同内容时拒绝覆盖。最后生成与公开工作区匹配的本地运行清单。所有导入材料均由 `.gitignore` 排除。

原包的严格校验针对当前既有队列；更换数据集需要按新的真实来源重新处理、核验并建立独立运行清单，不能通过修改哈希绕过校验。

## 双 GPU 服务器运行

```bash
bash run_smoke.sh smoke_v173
bash run_server_dual_gpu.sh server_v173

# 回传完整结果，包括模型、预测、环境与日志
tar -czf results_v173_server.tar.gz server_v173
```

默认 GPU0 负责外层折0/2/4，GPU1负责1/3；种子为20261023/24/25，分别用于折内初始化集成。运行目录必须为新目录。可使用 `FOUNDATION=0` 省略 TabPFN，或在协议和代码未变时使用 `RESUME=1` 跳过已完成的折；未完成折不会被自动覆盖。

```bash
FOUNDATION=0 bash run_server_dual_gpu.sh server_v173_no_foundation
RESUME=1 bash run_server_dual_gpu.sh server_v173
```

结果说明、各目标指标、预声明配对比较及重放记录位于 `server_v173/evaluation/`。无目标文件推理示例：

```bash
python -B code/infer_v173.py \
  --data /data/private/new_features --extras /data/private/new_extras \
  --model /data/private/final_models/residual_mmpareto.joblib \
  --output /data/private/predictions.csv --device cuda:0
```

新样本仍需满足相同特征定义和目标结构；默认拒绝对训练受试者或同组对象作独立推理。

## 研究边界

本项目研究量表评分预测，不能据此声称疾病诊断能力。既有队列和划分已经多轮用于开发，结果仍为探索性比较；三种子不是三套独立交叉验证。固定 OOF 预测的配对区间不覆盖历史模型搜索和重新训练的不确定性。

原始信号为非同步采集；未经确认的身份、访视、眼动范式和心率单位继续保留未确认状态。无依据的标签、身份确认或量表条目不应补造。心率单位未确认前，不将相关特征解释为标准 ms/Hz HRV。

TabPFN 在同服务器环境内可重放，但异平台/异设备仍出现数值差异；FP32 和输入快照是审计措施，不是跨平台一致性的保证。单个模块是否带来独立增益，需要匹配消融和后续独立验证。

## 代码与许可证

原数值代码、原服务器脚本和依赖文件按字节保留，哈希见 `PUBLIC_SOURCE_SHA256.json`；新增公开测试和私有运行材料导入工具不更改训练实现。

CBraMod 第三方代码的来源、固定提交、修改记录和 MIT 许可证保留于 `code/vendor_cbramod/`。TabPFN 通过独立 Python 包调用，权重未重新分发，使用时遵守相应原作者条款。
