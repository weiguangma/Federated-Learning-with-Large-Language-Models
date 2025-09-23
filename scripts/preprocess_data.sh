#!/bin/bash
# /********************************************************************************
#  * @Author: zhangqiuhong
#  * @Date: 2025-09-20
#  * @Description: 联邦学习数据预处理脚本
#  ********************************************************************************/

set -e  # 遇到错误时退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="/root/autodl-tmp/Federated_learning/code_v01/verify_data"
OUTPUT_DIR="$PROJECT_DIR/data/enhanced_qwen"
# OUTPUT_DIR="$PROJECT_DIR/data/debug_qwen"
SAMPLE_SIZE=5000
TRAIN_RATIO=0.8
RANDOM_SEED=42
RAILWAY_RATIO=0.6  # 铁路运输比例

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "联邦学习数据预处理脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -d, --data-dir DIR     原始数据目录 (默认: $DATA_DIR)"
    echo "  -o, --output-dir DIR   输出目录 (默认: $OUTPUT_DIR)"
    echo "  -s, --sample-size N    生成样本数量 (默认: $SAMPLE_SIZE)"
    echo "  -r, --train-ratio R    训练集比例 (默认: $TRAIN_RATIO)"
    echo "  --railway_ratio R     铁路运输比例 (默认: $RAILWAY_RATIO)"
    echo "  --seed N              随机种子 (默认: $RANDOM_SEED)"
    echo "  -h, --help            显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认参数"
    echo "  $0 -s 5000                           # 生成5000个样本"
    echo "  $0 -r 0.9                            # 90%训练集，10%测试集"
    echo "  $0 -d /path/to/data -o ./output      # 指定输入输出目录"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -r|--train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --railway_ratio)
            RAILWAY_RATIO="$2"
            shift 2
            ;;
        --seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主函数
main() {
    log_info "🚀 开始联邦学习数据预处理..."
    echo "=========================================="
    log_info "配置参数:"
    log_info "  - 原始数据目录: $DATA_DIR"
    log_info "  - 输出目录: $OUTPUT_DIR"
    log_info "  - 样本数量: $SAMPLE_SIZE"
    log_info "  - 训练集比例: $TRAIN_RATIO"
    log_info "  - 铁路运输比例: $RAILWAY_RATIO"
    log_info "  - 随机种子: $RANDOM_SEED"
    echo "=========================================="

    # 检查原始数据目录
    if [[ ! -d "$DATA_DIR" ]]; then
        log_error "原始数据目录不存在: $DATA_DIR"
        exit 1
    fi

    # 检查必要的数据文件（增强版只需要主数据文件）
    required_files=(
        "潜在箱源_海关模拟_样例全.csv"
    )

    log_info "📋 检查数据文件..."
    for file in "${required_files[@]}"; do
        if [[ ! -f "$DATA_DIR/$file" ]]; then
            log_error "缺少必要数据文件: $file"
            exit 1
        else
            log_success "✓ $file"
        fi
    done

    # 创建输出目录
    log_info "📁 创建输出目录..."
    mkdir -p "$OUTPUT_DIR"

    # 切换到项目目录
    cd "$PROJECT_DIR"

    # 检查Python环境
    if ! command -v python &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        exit 1
    fi

    # 检查增强版数据处理脚本
    if [[ ! -f "enhanced_data_processor.py" ]]; then
        log_error "增强版数据处理脚本不存在: enhanced_data_processor.py"
        exit 1
    fi

    # 执行增强版数据预处理
    log_info "🔧 执行增强版数据预处理..."
    python enhanced_data_processor.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --sample_size "$SAMPLE_SIZE" \
        --train_ratio "$TRAIN_RATIO" \
        --railway_ratio "$RAILWAY_RATIO" \
        --random_seed "$RANDOM_SEED"

    if [[ $? -eq 0 ]]; then
        log_success "🎉 数据预处理完成！"
        
        # 显示结果信息
        if [[ -f "$OUTPUT_DIR/enhanced_qwen_train.jsonl" ]]; then
            train_size=$(du -h "$OUTPUT_DIR/enhanced_qwen_train.jsonl" | cut -f1)
            train_count=$(wc -l < "$OUTPUT_DIR/enhanced_qwen_train.jsonl")
            log_success "训练集: $OUTPUT_DIR/enhanced_qwen_train.jsonl ($train_size, $train_count 样本)"
        fi
        
        if [[ -f "$OUTPUT_DIR/enhanced_qwen_test.jsonl" ]]; then
            test_size=$(du -h "$OUTPUT_DIR/enhanced_qwen_test.jsonl" | cut -f1)
            test_count=$(wc -l < "$OUTPUT_DIR/enhanced_qwen_test.jsonl")
            log_success "测试集: $OUTPUT_DIR/enhanced_qwen_test.jsonl ($test_size, $test_count 样本)"
        fi
        
        if [[ -f "$OUTPUT_DIR/enhanced_processing_stats.json" ]]; then
            log_success "统计信息: $OUTPUT_DIR/enhanced_processing_stats.json"
        fi
        
        echo "=========================================="
        log_info "✅ 预处理完成，可以开始训练："
        log_info "   bash scripts/run_train.sh"
        echo "=========================================="
    else
        log_error "❌ 数据预处理失败！"
        exit 1
    fi
}

# 运行主函数
main "$@"
