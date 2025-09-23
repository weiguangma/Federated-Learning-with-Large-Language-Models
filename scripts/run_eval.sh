#!/bin/bash
# /********************************************************************************
#  * @Author: zhangqiuhong
#  * @Date: 2025-09-20
#  * @Description: 联邦学习模型评估脚本
#  ********************************************************************************/

set -e  # 遇到错误时退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct"
CHECKPOINT_PATH="$PROJECT_DIR/federated_qwen_output/federated_qwen_model.pth"
TEST_DATA="$PROJECT_DIR/data/enhanced_qwen/enhanced_qwen_test.jsonl"
OUTPUT_FILE="$PROJECT_DIR/evaluation_results.json"
MAX_SAMPLES=10000
MAX_NEW_TOKENS=200
TEMPERATURE=0.9
DEVICE="cuda"

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
    echo "联邦学习模型评估脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model-path PATH         预训练模型路径 (默认: $MODEL_PATH)"
    echo "  -c, --checkpoint-path PATH    训练后的检查点路径 (默认: $CHECKPOINT_PATH)"
    echo "  -t, --test-data PATH         测试数据文件 (默认: $TEST_DATA)"
    echo "  -o, --output-file PATH       评估结果输出文件 (默认: $OUTPUT_FILE)"
    echo "  -n, --max_samples N          最大测试样本数 (默认: $MAX_SAMPLES)"
    echo "  --max-new-tokens N           生成的最大token数 (默认: $MAX_NEW_TOKENS)"
    echo "  --temperature T              生成温度 (默认: $TEMPERATURE)"
    echo "  --device DEVICE              计算设备 (默认: $DEVICE)"
    echo "  --quick                      快速评估模式 (50个样本)"
    echo "  --full                       完整评估模式 (500个样本)"
    echo "  --detailed                   详细评估模式 (包含更多指标)"
    echo "  -h, --help                  显示帮助信息"
    echo ""
    echo "预设模式:"
    echo "  --quick: 快速评估 (50个样本，温度0.5)"
    echo "  --full:  完整评估 (500个样本，温度0.7)"
    echo "  --detailed: 详细评估 (100个样本，更多生成token)"
    echo ""
    echo "示例:"
    echo "  $0                                           # 使用默认参数"
    echo "  $0 --quick                                   # 快速评估模式"
    echo "  $0 --full                                    # 完整评估模式"
    echo "  $0 -c ./my_model.pth -n 200                # 自定义检查点和样本数"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--checkpoint-path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        -t|--test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        -o|--output-file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -n|--max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --quick)
            MAX_SAMPLES=50
            TEMPERATURE=0.5
            MAX_NEW_TOKENS=150
            OUTPUT_FILE="$PROJECT_DIR/quick_evaluation_results.json"
            log_info "🚀 启用快速评估模式"
            shift
            ;;
        --full)
            MAX_SAMPLES=500
            TEMPERATURE=0.7
            MAX_NEW_TOKENS=250
            OUTPUT_FILE="$PROJECT_DIR/full_evaluation_results.json"
            log_info "🏁 启用完整评估模式"
            shift
            ;;
        --detailed)
            MAX_SAMPLES=100
            TEMPERATURE=0.7
            MAX_NEW_TOKENS=300
            OUTPUT_FILE="$PROJECT_DIR/detailed_evaluation_results.json"
            log_info "🔍 启用详细评估模式"
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
    if [[ "$DEVICE" == "cuda" ]]; then
        if command -v nvidia-smi &> /dev/null; then
            gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1)
            log_success "🎮 GPU可用: $gpu_info"
            
            # 检查GPU内存
            free_memory=$(echo "$gpu_info" | cut -d',' -f3 | tr -d ' ')
            if [[ $free_memory -lt 4000 ]]; then
                log_warning "⚠️  GPU内存不足4GB，建议使用CPU评估"
                DEVICE="cpu"
            fi
        else
            log_warning "⚠️  未检测到GPU，切换到CPU评估"
            DEVICE="cpu"
        fi
    else
        log_info "🖥️  使用CPU进行评估"
    fi
}

# 寻找最新的检查点
find_latest_checkpoint() {
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        log_warning "指定的检查点不存在: $CHECKPOINT_PATH"
        log_info "🔍 搜索可用的检查点..."
        
        # 搜索可能的检查点文件
        possible_paths=(
            "$PROJECT_DIR/federated_qwen_output/federated_qwen_model.pth"
            "$PROJECT_DIR/quick_train_output/federated_qwen_model.pth"
            "$PROJECT_DIR/full_train_output/federated_qwen_model.pth"
            "$PROJECT_DIR"/*_output/federated_qwen_model.pth
        )
        
        for path in "${possible_paths[@]}"; do
            if [[ -f "$path" ]]; then
                CHECKPOINT_PATH="$path"
                log_success "找到检查点: $CHECKPOINT_PATH"
                break
            fi
        done
        
        if [[ ! -f "$CHECKPOINT_PATH" ]]; then
            log_error "未找到可用的模型检查点"
            log_info "请先运行训练: bash scripts/run_train.sh"
            exit 1
        fi
    fi
}

# 主函数
main() {
    log_info "🧪 开始联邦学习模型评估..."
    echo "=========================================="
    log_info "配置参数:"
    log_info "  - 模型路径: $MODEL_PATH"
    log_info "  - 检查点路径: $CHECKPOINT_PATH"
    log_info "  - 测试数据: $TEST_DATA"
    log_info "  - 输出文件: $OUTPUT_FILE"
    log_info "  - 最大样本数: $MAX_SAMPLES"
    log_info "  - 最大生成token: $MAX_NEW_TOKENS"
    log_info "  - 生成温度: $TEMPERATURE"
    log_info "  - 计算设备: $DEVICE"
    echo "=========================================="

    # 检查模型路径
    if [[ ! -d "$MODEL_PATH" ]]; then
        log_error "预训练模型路径不存在: $MODEL_PATH"
        exit 1
    fi

    # 寻找检查点
    find_latest_checkpoint

    # 检查测试数据
    if [[ ! -f "$TEST_DATA" ]]; then
        log_error "测试数据文件不存在: $TEST_DATA"
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

    # 检查评估脚本
    if [[ ! -f "evaluate.py" ]]; then
        log_error "评估脚本不存在: evaluate.py"
        exit 1
    fi

    # 显示测试数据信息
    if [[ -f "$TEST_DATA" ]]; then
        file_size=$(du -h "$TEST_DATA" | cut -f1)
        line_count=$(wc -l < "$TEST_DATA")
        actual_samples=$((line_count < MAX_SAMPLES ? line_count : MAX_SAMPLES))
        log_info "📊 测试数据: $line_count 个样本 ($file_size)"
        log_info "📊 实际评估: $actual_samples 个样本"
    fi

    # 显示模型信息
    if [[ -f "$CHECKPOINT_PATH" ]]; then
        model_size=$(du -h "$CHECKPOINT_PATH" | cut -f1)
        log_info "🤖 模型大小: $model_size"
    fi

    # 构建评估命令
    eval_cmd="python evaluate.py \
        --model_path \"$MODEL_PATH\" \
        --checkpoint_path \"$CHECKPOINT_PATH\" \
        --test_data \"$TEST_DATA\" \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE \
        --output_file \"$OUTPUT_FILE\" \
        --device \"$DEVICE\""

    # 执行评估
    log_info "🔬 开始模型评估..."
    log_info "💡 命令: $eval_cmd"
    echo "=========================================="
    
    eval $eval_cmd

    if [[ $? -eq 0 ]]; then
        log_success "🎉 模型评估完成！"
        
        # 显示结果信息
        if [[ -f "$OUTPUT_FILE" ]]; then
            result_size=$(du -h "$OUTPUT_FILE" | cut -f1)
            log_success "评估结果: $OUTPUT_FILE ($result_size)"
            
            # 尝试显示关键指标
            if command -v jq &> /dev/null && [[ -f "$OUTPUT_FILE" ]]; then
                echo "=========================================="
                log_info "📈 关键评估指标:"
                
                transport_acc=$(jq -r '.metrics.transport_accuracy // "N/A"' "$OUTPUT_FILE" 2>/dev/null)
                city_acc=$(jq -r '.metrics.city_accuracy // "N/A"' "$OUTPUT_FILE" 2>/dev/null)
                bleu_score=$(jq -r '.metrics.avg_bleu_score // "N/A"' "$OUTPUT_FILE" 2>/dev/null)
                success_rate=$(jq -r '.metrics.generation_success_rate // "N/A"' "$OUTPUT_FILE" 2>/dev/null)
                
                if [[ "$transport_acc" != "N/A" ]]; then
                    transport_pct=$(echo "$transport_acc * 100" | bc -l 2>/dev/null | cut -d'.' -f1 2>/dev/null || echo "N/A")
                    log_info "  - 运输方式准确率: ${transport_pct}%"
                fi
                
                if [[ "$city_acc" != "N/A" ]]; then
                    city_pct=$(echo "$city_acc * 100" | bc -l 2>/dev/null | cut -d'.' -f1 2>/dev/null || echo "N/A")
                    log_info "  - 目的地准确率: ${city_pct}%"
                fi
                
                if [[ "$bleu_score" != "N/A" ]]; then
                    log_info "  - 平均BLEU分数: $bleu_score"
                fi
                
                if [[ "$success_rate" != "N/A" ]]; then
                    success_pct=$(echo "$success_rate * 100" | bc -l 2>/dev/null | cut -d'.' -f1 2>/dev/null || echo "N/A")
                    log_info "  - 生成成功率: ${success_pct}%"
                fi
            fi
        fi
        
        echo "=========================================="
        log_info "✅ 评估完成！查看详细结果:"
        log_info "   cat \"$OUTPUT_FILE\""
        echo "=========================================="
    else
        log_error "❌ 模型评估失败！"
        exit 1
    fi
}

# 运行主函数
main "$@"
