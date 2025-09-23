#!/bin/bash
# /********************************************************************************
#  * @Author: zhangqiuhong
#  * @Date: 2025-09-20
#  * @Description: 增强版联邦学习训练示例脚本
#  ********************************************************************************/

set -e  # 遇到错误时退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 增强版联邦学习训练示例${NC}"
echo "=========================================="

# 示例1：快速测试（小数据集，平衡运输方式）
echo -e "${GREEN}示例1: 快速测试训练${NC}"
echo "数据预处理 + 训练（100样本，50%铁路运输）"
echo "命令："
echo "  bash scripts/preprocess_data.sh -s 100 --railway_ratio 0.5"
echo "  bash scripts/run_train.sh --quick"
echo ""

# 示例2：中等规模训练
echo -e "${GREEN}示例2: 中等规模训练${NC}"
echo "数据预处理 + 训练（2000样本，60%铁路运输）"
echo "命令："
echo "  bash scripts/preprocess_data.sh -s 2000 --railway_ratio 0.6"
echo "  bash scripts/run_train.sh --epochs 5 --batch-size 4"
echo ""

# 示例3：大规模训练
echo -e "${GREEN}示例3: 大规模训练${NC}"
echo "数据预处理 + 训练（10000样本，70%铁路运输）"
echo "命令："
echo "  bash scripts/preprocess_data.sh -s 10000 --railway_ratio 0.7"
echo "  bash scripts/run_train.sh --full"
echo ""

# 示例4：自定义运输方式比例
echo -e "${GREEN}示例4: 自定义运输方式比例${NC}"
echo "不同运输方式比例的训练对比"
echo "命令："
echo "  # 80%铁路运输（偏向铁路）"
echo "  bash scripts/preprocess_data.sh -s 1000 --railway_ratio 0.8 -o ./data/railway_heavy"
echo ""
echo "  # 20%铁路运输（偏向公路）"
echo "  bash scripts/preprocess_data.sh -s 1000 --railway_ratio 0.2 -o ./data/road_heavy"
echo ""

# 示例5：完整的训练流程
echo -e "${GREEN}示例5: 完整的训练+评估流程${NC}"
echo "从数据预处理到模型评估的完整流程"
echo "命令："
echo "  # 1. 数据预处理"
echo "  bash scripts/preprocess_data.sh -s 5000 --railway_ratio 0.6"
echo ""
echo "  # 2. 模型训练"
echo "  bash scripts/run_train.sh --epochs 10 --learning-rate 1e-6"
echo ""
echo "  # 3. 模型评估"
echo "  bash scripts/run_eval.sh --max_samples 200"
echo ""

echo "=========================================="
echo -e "${YELLOW}💡 提示：${NC}"
echo "1. 运输方式比例控制："
echo "   --railway_ratio 0.6  # 60%铁路，40%公路"
echo "   --railway_ratio 0.3  # 30%铁路，70%公路"
echo ""
echo "2. 训练模式选择："
echo "   --quick     # 快速测试（小batch，少epoch）"
echo "   --full      # 完整训练（大batch，多epoch）"
echo ""
echo "3. 数据质量保证："
echo "   - 自动过滤无效数据"
echo "   - 港口、铁路、海关信息完整"
echo "   - 目的地信息真实有效"
echo ""
echo "4. Split Learning架构："
echo "   - 客户端只做embedding前向传播"
echo "   - 服务端完整模型计算和权重更新"
echo "   - 自动同步embedding权重到客户端"
echo "=========================================="
