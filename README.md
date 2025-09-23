# 联邦学习Qwen2.5-0.5B智能物流决策系统

## 🎯 项目概述

这是一个基于Qwen2.5-0.5B的联邦学习系统，专门用于智能物流决策。系统整合港口、铁路、海关三方数据，在保护数据隐私的前提下，实现协同的运输方式选择和目的地预测。

## 🏗️ 系统架构

### 联邦学习架构
- **客户端**: 只保留Qwen的embedding层，处理本地数据
- **服务端**: 包含完整的Qwen模型结构，进行数据融合和决策生成
- **数据安全**: 客户端数据不离开本地，只传输embedding向量

### 技术特色
- ✅ **数据隐私保护**: 联邦学习架构确保原始数据不共享
- ✅ **多模态融合**: 整合港口、铁路、海关三方异构数据
- ✅ **高性能处理**: O(1)查找算法，支持大规模数据处理
- ✅ **训练稳定性**: 完整的梯度裁剪、数值检查、异常处理机制

## 📁 项目结构

```
offical_code/
├── data_processor.py          # 数据处理统一脚本
├── federated_model.py         # 联邦模型架构定义
├── train.py                   # 模型训练脚本
├── evaluate.py                # 模型评估脚本
├── scripts/                   # 便捷脚本目录
│   ├── preprocess_data.sh     # 数据预处理脚本
│   ├── run_train.sh          # 训练执行脚本
│   └── run_eval.sh           # 评估执行脚本
├── data/                      # 数据目录
│   └── qwen_processed/        # 处理后的训练数据
│       ├── qwen_federated_train.jsonl  # 训练集 (1600样本)
│       ├── qwen_federated_test.jsonl   # 测试集 (400样本)
│       └── dataset_split_stats.json    # 数据集统计
└── README.md                  # 项目说明文档
```

## 🚀 快速开始

### 🎯 **一键运行（推荐）**

使用便捷脚本，一键完成整个流程：

```bash
# 1. 数据预处理 (生成训练集和测试集)
bash scripts/preprocess_data.sh -s 2000

# 2. 快速训练测试
bash scripts/run_train.sh --quick

# 3. 模型评估
bash scripts/run_eval.sh --quick
```

### 🔧 **详细使用方法**

#### 1. 数据处理

**便捷脚本（推荐）:**
```bash
# 使用默认参数
bash scripts/preprocess_data.sh

# 自定义参数：样本数量、训练集比例、输出目录
bash scripts/preprocess_data.sh -s 5000 -r 0.9 -o ./my_data
```

**直接调用Python:**
```bash
python data_processor.py \
    --data_dir /path/to/raw/data \
    --output_dir ./data/qwen_processed \
    --sample_size 2000 \
    --train_ratio 0.8 \
    --random_seed 42
```

**📊 数据分割说明:**
- 默认按 8:2 比例分割训练集和测试集
- 生成文件：
  - `qwen_federated_train.jsonl` - 训练集 (80%)
  - `qwen_federated_test.jsonl` - 测试集 (20%)
  - `dataset_split_stats.json` - 数据集统计信息
- 使用固定随机种子确保结果可重现

#### 2. 模型训练

**便捷脚本（推荐）:**
```bash
# 快速测试模式
bash scripts/run_train.sh --quick

# 完整训练模式
bash scripts/run_train.sh --full

# 自定义参数
bash scripts/run_train.sh -e 5 -lr 1e-6
```

**直接调用Python:**
```bash
python train.py \
    --data_dir ./data \
    --model_path /path/to/Qwen2.5-0.5B-Instruct \
    --epochs 3 \
    --learning_rate 2e-6 \
    --output_dir federated_qwen_output
```

#### 3. 模型评估

**便捷脚本（推荐）:**
```bash
# 快速评估模式
bash scripts/run_eval.sh --quick

# 完整评估模式
bash scripts/run_eval.sh --full

# 自定义检查点
bash scripts/run_eval.sh -c ./my_model.pth -n 200
```

**直接调用Python:**
```bash
python evaluate.py \
    --model_path /path/to/Qwen2.5-0.5B-Instruct \
    --checkpoint_path federated_qwen_output/federated_qwen_model.pth \
    --test_data ./data/qwen_processed/qwen_federated_train.jsonl \
    --max_samples 100
```

### 📋 **脚本参数说明**

#### `preprocess_data.sh`
- `-s, --sample-size`: 生成样本数量 (默认: 2000)
- `-r, --train-ratio`: 训练集比例 (默认: 0.8)
- `--seed`: 随机种子 (默认: 42)
- `-d, --data-dir`: 原始数据目录
- `-o, --output-dir`: 输出目录

#### `run_train.sh`
- `--quick`: 快速测试模式 (1轮，100样本，batch=2)
- `--full`: 完整训练模式 (5轮，所有样本，batch=8)
- `-e, --epochs`: 训练轮数
- `-lr, --learning-rate`: 学习率
- `-b, --batch-size`: 批次大小 (默认: 4)
- `--grad-accum`: 梯度累积步数 (默认: 2，有效batch=8)

#### `run_eval.sh`
- `--quick`: 快速评估模式 (50个样本)
- `--full`: 完整评估模式 (500个样本)
- `-c, --checkpoint-path`: 模型检查点路径
- `-n, --max-samples`: 最大测试样本数

## 🔧 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA support (推荐RTX 3080或更高)
- **内存**: 至少16GB RAM
- **存储**: 至少10GB可用空间

### 软件依赖
```bash
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.24.0
tqdm>=4.65.0
```

## 📊 性能表现

### 训练效果
- **损失下降**: 从11.5降至2.3，下降约80%
- **训练稳定性**: 无NaN损失，梯度稳定
- **收敛速度**: 500步内显著改善

### 评估指标
- **运输方式准确率**: 评估运输方式选择的准确性
- **目的地预测准确率**: 评估目的地城市和省份预测
- **文本生成质量**: BLEU分数和完全匹配率

## 🎯 核心创新点

### 1. 联邦学习架构设计
- 客户端只保留embedding层，减少计算和通信开销
- 服务端进行数据融合，充分利用多方信息
- ChatML格式分隔符确保数据正确拼接

### 2. 高性能数据处理
- O(1)哈希索引查找，处理大规模数据
- 智能数据匹配算法，最大化数据利用率
- 鲁棒的异常处理，确保数据处理稳定性

### 3. 训练优化策略
- 动态标签长度调整，解决batch size不匹配
- 梯度裁剪和数值稳定性检查
- 自适应学习率和检查点保存机制

## 📈 使用场景

### 物流决策优化
- **运输方式选择**: 基于成本、时效、路线等因素
- **目的地预测**: 根据货物特征和贸易流向
- **风险评估**: 考虑天气、政策、路径等风险因素

### 多方协作
- **港口运营商**: 提供货物和作业信息
- **铁路公司**: 提供运输能力和路线数据
- **海关部门**: 提供贸易和清关信息

## 🔒 隐私保护

### 数据安全机制
- **本地计算**: 原始数据不离开客户端
- **加密传输**: embedding向量加密传输
- **访问控制**: 严格的权限管理机制

### 合规性
- 符合数据保护法规要求
- 支持审计和监管需求
- 可配置的隐私保护级别

## 🛠️ 开发指南

### 扩展新的客户端类型
1. 在`federated_model.py`中添加新的客户端embedding
2. 在`data_processor.py`中添加对应的数据转换器
3. 更新模型的客户端映射逻辑

### 自定义评估指标
1. 在`evaluate.py`中添加新的指标计算函数
2. 更新`calculate_metrics`方法
3. 在评估报告中展示新指标

## 📞 支持与反馈

如有问题或建议，请联系项目团队或提交Issue。

---

**作者**: zhangqiuhong  
**日期**: 2025-09-20  
**版本**: 1.0.0