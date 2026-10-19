# V8：小样本 EEG / 眼动改进包

用于个人级 SCL-90 抑郁、焦虑及 PSQI 分数回归。不是医生诊断分类，也不承诺高指标。
原始信号、V3 特征、V7 代码和结果只读；所有处理结果进入新目录，已存在的输出目录会被拒绝。

## 实际修改

1. **保留区域信息**：官方预训练 CBraMod 冻结输出，先平均时间 patch，再按额、中央、颞、顶、枕五个固定电极区域聚合。每人窗口均值和标准差共 2000 维；原来全通道平均的 400 维同时导出作一致性对照。五区域覆盖现有 16 电极，无重复，不推断睁闭眼等事件语义。
2. **控制过拟合**：区域频带特征从 160 维归纳为 50 维，CBraMod 的区域特征降为 8/16 主成分。中位数填补、标准化、PCA 白化、目标标准化均只拟合当前训练受试者；Ridge alpha 固定为 10/100/1000。
3. **个人级后期融合**：分别得到 EEG 和眼动预测，在内层验证上共同选择分支参数和 EEG 权重 0.25/0.5/0.75，保留两个单模态端点。绝不按窗口编号对齐不同时间采集的信号；不直接拼接 2000+20 维后用相同列惩罚强迫融合。
4. **显式回退**：guarded_fusion 把训练均值/中位数也作为候选，采用一标准误差启发式优先选择更简单的候选。不回退的 late_fusion 单独报告。三折标准误差仅是保守选择启发式，不能解释为统计置信区间。
5. **人口学对照**：年龄/性别仅进入 demographics_control，不进入其余五项生理信号候选。人口学模型好，不等于脑电/眼动有效。

所有网格、选择规则与代码哈希在计算外层分数前记录在 protocol.json。没有按外层成绩删样本、换划分、修改标签或截断预测。
V8 基于已观察的 V7 结果设计；复用外层划分属于开发期比较，不能充当未触碰过的最终测试集。

## 六组结果

| 名称 | 含义 | 选择规则 |
|---|---|---|
| power_pca | 五区域频带、训练集 PCA、Ridge | 内层 MAE，一标准误差 |
| regional_cbramod | 五区域冻结 CBraMod、训练集 PCA、Ridge | 内层 MAE，一标准误差 |
| eye_regularized | 现有 20 维眼动统计，SVR C=0.1/1 | 内层 MAE，一标准误差 |
| late_fusion | 上述分支的固定权重预测融合，包含单模态端点 | 最低内层 MAE，无常数候选 |
| guarded_fusion | 同一生理候选池加训练均值/中位数 | 内层 MAE，一标准误差，优先简单模型 |
| demographics_control | 年龄、性别 Ridge，对照用途 | 内层 MAE，一标准误差 |

EEG 两分支各 6 个配置，眼动 2 个，融合 72 个固定组合；这些组合复用内层留出预测，不重复训练基础分支。guarded_fusion 共 88 个候选，候选池较大仍有选择方差；故同时提供单模态结果和描述性配对区间。
一标准误差内优先级：常数、眼动、频带 EEG、CBraMod EEG、双分支；同类优先更强正则和更少主成分。均值/中位数之间按内层 MAE 选。

## 环境

本地已验证 Python 3.11、torch 2.5.1+cu121、NumPy 1.26.4、scikit-learn 1.5.2、SciPy 1.13.1、joblib 1.4.2。
建议新建环境；PyTorch CUDA 版本需匹配服务器驱动。本版不需要安装或运行 TabPFN。

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_v8.txt
python -B verify_package.py
python -B -m unittest test_core test_baseline test_improved -v
```

包中保留 V7 兼容脚本、官方权重和来源/许可证，用于复核；V8 入口是 build_improved.py，不是 build_baseline.py。不开启云端推理或在线下载权重。

## 使用已有特征和冻结划分（推荐）

另行上传以下已有文件，不把原始患者数据打进代码压缩包：

- V3 的完整 features 目录（含 manifest.json、config.json、review_snapshot.json 和所有 NPZ）。
- 原情绪 plan.json（86 人），原 PSQI plan.json（80 人）。不要重新随机分组。
- 迁移需要的原始/processed 数据目录；若选择重新处理原始数据，也可用下文 raw-root 入口。

Windows 的绝对路径不能直接用于 Linux。只对服务器上复制的特征配置迁移，manifest 和 NPZ 完全保持一致：

```bash
python -B relocate_features.py --source /data/uploaded_features \
  --output /data/features_linux --raw-root /data/raw --processed-root /data/processed
bash run_v8.sh /data/features_linux /data/emotion/plan.json /data/psqi/plan.json /results/v8_run_01
```

以上路径是示例，替换为实际位置。raw/processed 路径应指向真实输入目录，且不与输出相互包含。
如果复制的 config.json 已包含当前系统正确路径，跳过 relocate_features.py。

显式命令：

```bash
python -B build_improved.py --features /data/features_linux \
  --plans /data/emotion/plan.json /data/psqi/plan.json \
  --output /results/v8_run_01 --devices cuda:0 cuda:1 \
  --batch-size 16 --threads 2 --allow-unreviewed
```

CBraMod 特征提取使用第一张 GPU。两个目标计划可并行运行，但其小样本回归头使用 CPU；本版不做 DDP，不会声称两张 GPU 都在训练大型网络。输出目录中可看到每个任务的独立日志。
--allow-unreviewed 只允许探索性分析，不等于确认了同访视、任务或诊断。

若已有本版 regional_embeddings，可加 --embeddings 指向该目录；V7 的 400 维目录不能冒充区域表征。每位受试者输入和输出 SHA256 都会检查。

## 从原始数据重新处理

```bash
python -B build_improved.py --raw-root /data/raw --output /results/v8_fresh_01 \
  --devices cuda:0 cuda:1 --batch-size 16 --allow-unreviewed
```

此入口会创建新的 processed、features 和计划；若数据/审核内容变化，就不能直接与原 V7 指标作配对比较。正式分析请先完成审核，再去掉 --allow-unreviewed。

## 输出与对比

主输出：status.json、regional_embeddings、plans、runs/task_0、runs/task_1。每个任务包含：

- protocol.json、selections.json：全部候选内层误差和最终决策，先于外层评估写入。
- predictions.csv、summary.json/md：原量表单位的个人级外层预测和指标。
- models：模型、训练/测试 ID、哈希、重载复测输入。
- verification.json：覆盖、group 隔离、指标重算和全部保存模型重载的结果。

与 V7 的完整匹配划分结果比较：

```bash
python -B compare_v7.py --old /results/v7/runs/task_0 /results/v7/runs/task_1 \
  --new /results/v8_run_01/runs/task_0 /results/v8_run_01/runs/task_1 \
  --output /results/v8_comparison_01
```

只有两版 plan 哈希一致、完整外层覆盖且标签一致才允许比较。报告不自动挑选“最好目标”。

## 仍然存在的数据限制

本版不改变原 EEG 滤波、参考、4 秒窗或幅值阈值；不再次 /100。没有根据外层误差剔除受试者。
眼动沿用现有统计，未确认的任务/坐标不用于追踪误差，不创造瞳孔字段；PPG/NNI 不强行加入。
同访视审核、EEG 事件含义、眼动任务差异、量表时效和新增受试者标签仍需补全。
若 R² 接近零、优势不稳定或多折回退常数，请据实认定当前信号尚不足，不能把保守回退包装成模型发现。

官方 CBraMod 代码及权重来源沿用 vendor_cbramod/provenance.json、weights 内的记录和许可证；未声称是官方下游微调方案的完整复现。
