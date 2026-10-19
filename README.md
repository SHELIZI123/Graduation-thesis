# Graduation-thesis

面向精神心理健康评估的脑电表征学习与多模态融合方法研究。

目前发布 [V10 源码与运行说明](v10/README.md)：受试者级 EEG / 眼动集合核、连接统计压缩及训练折内凸组合融合。

## 快速开始

Python 3.11，PyTorch 2.5.1；在本项目既有 CUDA 12.1 环境中可按以下方式安装：

```bash
cd v10
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m unittest test_literature -v
python prepare_runtime.py --source /data/private/eeg_eye_v10_literature_20261016
```

训练命令、路径迁移和独立复核见 V10 README。核模型训练使用 CPU；重新提取冻结 CBraMod 特征才需要 GPU，当前实现没有双卡 DDP。

## 数据与结果

公开仓库不含患者原始信号、受试者清单、量表、划分配置、特征缓存、逐人预测或训练产物。复跑本队列需要通过私有途径取得原 V10 包中的运行时配置和已有数据。`prepare_runtime.py` 保留原配置、权重及引擎的 SHA256 校验。

历史开发折被重复用于版本比较，结果仅作探索性分析；尚不能作为独立测试、诊断性能或创新有效性的证明。原始数据不在此仓库内，所有新运行必须输出至新的目录。

第三方代码、权重的许可证和来源保留在 `v10/engine/vendor_cbramod/` 与 `v10/engine/weights/`。
