# V10：文献驱动的EEG与眼动受试者级模型

独立于V8/V9的新版本。情绪86人、PSQI80人的历史开发集比较；不是独立最终测试。原始数据、量表、人工确认列和旧实验均不修改。无真实标签的人不补造量表；reserve候选175人中训练使用0人。

主要新实现位于 literature_models.py：精确均值嵌入集合核、传感器连接压缩、训练折内的凸组合集成。预设主模型 fusion_stack；集合核融合 set_fusion 是次要候选。不能事后互换主次地位。

## 历史开发结果


集合核融合的OOF MAE：抑郁0.7229、焦虑0.7405、PSQI3.1586。对应R²为0.0567、0.1480、-0.0765。加权集成主模型MAE为0.7454、0.8021、3.0230，没有一致提升。严格FDR连接压缩均退化为均值预测。条件bootstrap优势区间仍跨零。不要把较好的次要候选数字解释为确认性检验或模型部署成绩。

## 依赖与输入

Python 3.11；安装 requirements.txt，并按服务器CUDA环境安装匹配的PyTorch。训练新核模型在CPU上运行；读取旧模块仍需要torch。重新提取CBraMod表征才需要GPU。未测试服务器双4090或DDP；本轮不需要两张GPU。

需要自行安全传输的私有输入：

1. 原特征目录 multimodal_v3_validation_20261013/features，包含manifest、config、review和全部261个NPZ，用于锁定与一致性核验。
2. 原处理目录 processed，以及配置指向的raw根目录。当前校验器要求这些路径存在。它们只读。
3. 新特征缓存 eeg_eye_v9_features_20261016（86个NPZ及features.json）。这是已保存的无标签特征，不是新造样本。

公开仓库不含私有信号、特征缓存、量表、拟合模型或 config。config 只在本地导入，并由 .gitignore 排除。代码中保留了旧支持模块用于读取和重建特征；本版本训练入口为run_literature.py，推理入口为infer_literature.py。

## GitHub 公开源码的准备步骤

模型源码与原 V10 包逐字节一致。私有配置、权重文件和一份包含本地路径的旧引擎记录不纳入 Git；原引擎完整校验仍要求这些运行时文件存在。请先把已拥有的 V10 独立包安全传至服务器、解压至仓库外，再执行：

```bash
python prepare_runtime.py --source /data/private/eeg_eye_v10_literature_20261016
```

该脚本先校验所有来源文件的 SHA256，仅补入缺少的运行时文件；已存在且不一致时拒绝覆盖，不修改原包、原数据或划分。权重从原包复制，不上传到 GitHub；第三方许可证与来源保留。此步骤后，再按下方迁移特征并训练。公开源码的数值测试可在不导入私有配置时运行。

## Linux路径迁移

先在服务器准备好raw和processed，再生成一份新的特征副本；只修改新副本的路径配置，保留原件。

```bash
python engine/relocate_features.py --source /data/features_original --output /data/features_linux --raw-root /data/raw --processed-root /data/processed
python -m unittest test_literature -v
python run_literature.py --features /data/features_linux --v9-features /data/eeg_eye_v9_features_20261016 --output /results/v10_new --threads 4
python verify_literature.py --features /data/features_linux --v9-features /data/eeg_eye_v9_features_20261016 --run /results/v10_new --output /results/v10_new/independent_verification.json
```

输出目录必须不存在；不会覆盖旧结果。不要编辑训练源代码、缓存或划分来绕过哈希保护。Windows上将路径替换为本地绝对路径即可。

## 无量表推理

输入为相同预处理结构的特征目录和脱敏ID，不需要问卷字段。模型输出只是预测分数，不是观测标签或诊断。若不传embeddings，代码会为指定ID新提取冻结CBraMod表征；若传入缓存，其manifest和记录哈希必须与输入目录匹配。

```bash
python infer_literature.py --features /data/future_features --run /results/v10_new --ids 301 302 --models set_fusion fusion_stack --output /results/future_prediction --device cuda:0
```

final下三个joblib是全开发集拟合模型；只加载自己生成且来源可信的joblib。推理输出会标记training_overlap。训练对象的重拟合预测不能用来报告泛化成绩。

## 实现和证据边界

- 所有窗口参与经验均值核；每人一个监督样本。训练尺度用每人20个固定位置估计以平衡长记录；带宽取训练样本距离。均值核保留对角项，外层高斯核按训练内嵌入距离定带宽；训练内核中心化。
- 连接输入195项：五频带各15个区域对传感器统计加120个矩阵对数协方差非对角项。压缩为六组正负共12维；FDR q<0.01与top16分开报告。top16不等于统计显著。
- 集成以固定基学习器的内层OOF预测拟合非负、和为1的权重，并固定0.02正则。没有外层调权重。它是Super Learner思路的适配，非原文完整复现。
- 超参使用3内层折内的one-SE启发式选择；该标准误不是泛化置信区间。最终核模型用五次内层选择的众数参数，集成最终权重用全开发集的固定基学习器折外预测拟合；不把最终模型训练误差作为结果。
- 原始EEG未在此轮增加ICA；眼动无瞳孔、未确认任务、不做时序对齐和目标误差推断。数字事件码不解释为已确认范式。
- 本轮未微调CBraMod、未实现新的门控注意力网络，也未复现源空间CPM。已知方法组合不是自动成立的论文原创贡献。

通过7个数值/边界测试，独立进程复算30组指标并回放120组外层模型输出；另外通过不含问卷字段的两人推理夹具检查（仅工程验证，两人均与训练重叠）。
