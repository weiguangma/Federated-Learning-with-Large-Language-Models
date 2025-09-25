#!/bin/bash

set -e  # 遇到错误时退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct"
DATA_DIR="$PROJECT_DIR/data/enhanced_qwen"
OUTPUT_DIR="$PROJECT_DIR/federated_qwen_output"
# DATA_DIR="$PROJECT_DIR/data/debug_qwen"
# OUTPUT_DIR="$PROJECT_DIR/debug_qwen_output"
EPOCHS=1
LEARNING_RATE=2e-6
BATCH_SIZE=8  # 提升到4，更好利用GPU
GRADIENT_ACCUMULATION_STEPS=2  # 梯度累积，等效batch_size=8
MAX_LENGTH=2048
SAVE_STEPS=200  # 减少保存频率，因为有效batch更大了
MAX_SAMPLES="10000"

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
    echo "联邦学习模型训练脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model-path PATH     预训练模型路径 (默认: $MODEL_PATH)"
    echo "  -d, --data-dir DIR        数据目录 (默认: $DATA_DIR)"
    echo "  -o, --output-dir DIR      输出目录 (默认: $OUTPUT_DIR)"
    echo "  -e, --epochs N           训练轮数 (默认: $EPOCHS)"
    echo "  -lr, --learning-rate LR   学习率 (默认: $LEARNING_RATE)"
    echo "  -b, --batch-size N       批次大小 (默认: $BATCH_SIZE)"
    echo "  --grad-accum N           梯度累积步数 (默认: $GRADIENT_ACCUMULATION_STEPS)"
    echo "  -l, --max-length N       最大序列长度 (默认: $MAX_LENGTH)"
    echo "  -s, --save-steps N       保存间隔 (默认: $SAVE_STEPS)"
    echo "  --max_samples N          最大样本数 (用于测试)"
    echo "  --quick                  快速训练模式 (1轮，100样本)"
    echo "  --full                   完整训练模式 (5轮，所有样本)"
    echo "  -h, --help              显示帮助信息"
    echo ""
    echo "预设模式:"
    echo "  --quick: 快速测试训练 (1轮，100样本，学习率5e-6)"
    echo "  --full:  完整正式训练 (5轮，所有样本，学习率1e-6)"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认参数"
    echo "  $0 --quick                           # 快速测试模式"
    echo "  $0 --full                            # 完整训练模式"
    echo "  $0 -e 5 -lr 1e-6                    # 自定义参数"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        -l|--max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        -s|--save-steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --quick)
            EPOCHS=1
            LEARNING_RATE=5e-6
            BATCH_SIZE=2  # 快速模式用较小batch
            GRADIENT_ACCUMULATION_STEPS=1
            MAX_SAMPLES=100
            SAVE_STEPS=50
            OUTPUT_DIR="$PROJECT_DIR/quick_train_output"
            log_info "🚀 启用快速训练模式"
            shift
            ;;
        --full)
            EPOCHS=5
            LEARNING_RATE=1e-6
            BATCH_SIZE=8  # 完整模式用更大batch
            GRADIENT_ACCUMULATION_STEPS=2  # 等效batch_size=16
            MAX_SAMPLES=""
            SAVE_STEPS=500
            OUTPUT_DIR="$PROJECT_DIR/full_train_output"
            log_info "🏁 启用完整训练模式"
            shift
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

# 检查GPU可用性
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1)
        log_success "🎮 GPU可用: $gpu_info"
        
        # 检查GPU内存
        free_memory=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
        if [[ $free_memory -lt 8000 ]]; then
            log_warning "⚠️  GPU内存不足8GB，建议释放内存或使用CPU训练"
        fi
    else
        log_warning "⚠️  未检测到GPU，将使用CPU训练（速度较慢）"
    fi
}

# 主函数
main() {
    log_info "🚀 开始联邦学习模型训练..."
    echo "=========================================="
    log_info "配置参数:"
    log_info "  - 模型路径: $MODEL_PATH"
    log_info "  - 数据目录: $DATA_DIR"
    log_info "  - 输出目录: $OUTPUT_DIR"
    log_info "  - 训练轮数: $EPOCHS"
    log_info "  - 学习率: $LEARNING_RATE"
    log_info "  - 批次大小: $BATCH_SIZE"
    log_info "  - 梯度累积: $GRADIENT_ACCUMULATION_STEPS 步 (有效batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)))"
    log_info "  - 最大长度: $MAX_LENGTH"
    log_info "  - 保存间隔: $SAVE_STEPS"
    if [[ -n "$MAX_SAMPLES" ]]; then
        log_info "  - 最大样本数: $MAX_SAMPLES"
    fi
    echo "=========================================="

    # 检查模型路径
    if [[ ! -d "$MODEL_PATH" ]]; then
        log_error "预训练模型路径不存在: $MODEL_PATH"
        exit 1
    fi

    # 检查数据目录和文件 - 支持增强版和旧版数据
    TRAIN_FILE_ENHANCED="$DATA_DIR/enhanced_qwen_train.jsonl"
    TRAIN_FILE_OLD="$DATA_DIR/qwen_processed/qwen_federated_train.jsonl"
    
    if [[ -f "$TRAIN_FILE_ENHANCED" ]]; then
        log_success "✓ 发现增强版训练数据: enhanced_qwen_train.jsonl"
    elif [[ -f "$TRAIN_FILE_OLD" ]]; then
        log_success "✓ 发现旧版训练数据: qwen_processed/qwen_federated_train.jsonl"
    else
        log_error "训练数据文件不存在，请检查以下路径之一:"
        log_error "  - $TRAIN_FILE_ENHANCED (增强版)"
        log_error "  - $TRAIN_FILE_OLD (旧版)"
        log_info "请先运行数据预处理: bash scripts/preprocess_data.sh"
        exit 1
    fi

    # 检查GPU
    check_gpu

    # 切换到项目目录
    cd "$PROJECT_DIR"

    # 检查Python环境
    if ! command -v python &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        exit 1
    fi

    # 检查训练脚本
    if [[ ! -f "train.py" ]]; then
        log_error "训练脚本不存在: train.py"
        exit 1
    fi

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"

    # 构建训练命令
    train_cmd="python train.py \
        --data_dir \"$DATA_DIR\" \
        --model_path \"$MODEL_PATH\" \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --max_length $MAX_LENGTH \
        --save_steps $SAVE_STEPS \
        --output_dir \"$OUTPUT_DIR\""

    if [[ -n "$MAX_SAMPLES" ]]; then
        train_cmd="$train_cmd --max_samples $MAX_SAMPLES"
    fi

    # 显示训练数据信息
    if [[ -f "$TRAIN_FILE_ENHANCED" ]]; then
        file_size=$(du -h "$TRAIN_FILE_ENHANCED" | cut -f1)
        line_count=$(wc -l < "$TRAIN_FILE_ENHANCED")
        log_info "📊 增强版训练数据: $line_count 个样本 ($file_size)"
    elif [[ -f "$TRAIN_FILE_OLD" ]]; then
        file_size=$(du -h "$TRAIN_FILE_OLD" | cut -f1)
        line_count=$(wc -l < "$TRAIN_FILE_OLD")
        log_info "📊 旧版训练数据: $line_count 个样本 ($file_size)"
    fi

    # 执行训练
    log_info "🔥 开始模型训练..."
    log_info "💡 命令: $train_cmd"
    echo "=========================================="
    
    eval $train_cmd

    if [[ $? -eq 0 ]]; then
        log_success "🎉 模型训练完成！"
        
        # 显示结果信息
        if [[ -f "$OUTPUT_DIR/federated_qwen_model.pth" ]]; then
            model_size=$(du -h "$OUTPUT_DIR/federated_qwen_model.pth" | cut -f1)
            log_success "模型文件: $OUTPUT_DIR/federated_qwen_model.pth ($model_size)"
        fi
        
        if [[ -f "$OUTPUT_DIR/training_config.json" ]]; then
            log_success "训练配置: $OUTPUT_DIR/training_config.json"
        fi
        
        echo "=========================================="
        log_info "✅ 训练完成，可以开始评估："
        log_info "   bash scripts/run_eval.sh --checkpoint_path \"$OUTPUT_DIR/federated_qwen_model.pth\""
        echo "=========================================="
    else
        log_error "❌ 模型训练失败！"
        exit 1
    fi
}

# 运行主函数
main "$@"
