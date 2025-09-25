#!/bin/bash

set -e  # 遇到错误立即退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ABLATION_CONFIG="$SCRIPT_DIR/ablation_configs.yaml"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
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

log_header() {
    echo -e "${PURPLE}[ABLATION]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
消融实验启动脚本

用法: $0 [选项]

选项:
    --experiment EXPERIMENT     运行指定的实验 (baseline|no_port|no_railway|no_customs|all)
    --epochs EPOCHS            训练轮数 (默认: 3)
    --learning-rate LR          学习率 (默认: 2e-6)
    --batch-size SIZE          批次大小 (默认: 4)
    --max-samples SAMPLES      最大样本数 (默认: 4000)
    --output-dir DIR           输出目录 (默认: ablation_experiments)
    --skip-data-generation     跳过数据生成步骤
    --dry-run                  只显示将要执行的命令，不实际运行
    --help                     显示此帮助信息

实验类型:
    baseline    - 完整联邦学习（港口+铁路+海关）
    no_port     - 去除港口客户端（铁路+海关）
    no_railway  - 去除铁路客户端（港口+海关）
    no_customs  - 去除海关客户端（港口+铁路）
    all         - 运行所有实验（默认）

示例:
    $0                                      # 运行所有消融实验
    $0 --experiment baseline               # 只运行基准实验
    $0 --experiment no_port --epochs 5    # 运行去除港口的实验，5个epoch
    $0 --dry-run                          # 预览所有将要执行的命令

EOF
}

# 默认参数
EXPERIMENT="all"
EPOCHS=3
LEARNING_RATE="2e-6"
BATCH_SIZE=4
MAX_SAMPLES=4000
OUTPUT_DIR="ablation_experiments"
SKIP_DATA_GEN=false
DRY_RUN=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-data-generation)
            SKIP_DATA_GEN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
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

# 验证实验类型
if [[ ! "$EXPERIMENT" =~ ^(baseline|no_port|no_railway|no_customs|all)$ ]]; then
    log_error "无效的实验类型: $EXPERIMENT"
    show_help
    exit 1
fi

# 确保在正确的目录中
cd "$PROJECT_ROOT"

# 创建输出目录结构
create_output_structure() {
    local base_dir="$1"
    log_info "创建输出目录结构: $base_dir"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$base_dir"/{data,models,logs,results,reports}
    else
        log_info "[DRY RUN] mkdir -p $base_dir/{data,models,logs,results,reports}"
    fi
}

# 生成数据（如果需要）
generate_data() {
    local clients="$1"
    local suffix="$2"
    
    if [[ "$SKIP_DATA_GEN" == "true" ]]; then
        log_info "跳过数据生成步骤"
        return 0
    fi
    
    log_info "为实验 $suffix 生成数据..."
    
    local data_output_dir="$OUTPUT_DIR/data/$suffix"
    local cmd="python enhanced_data_processor.py \
        --data_dir \"/root/autodl-tmp/Federated_learning/code_v01/verify_data\" \
        --output_dir \"$data_output_dir\" \
        --sample_size $MAX_SAMPLES \
        --railway_ratio 0.6 \
        --random_seed 42"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] $cmd"
    else
        if ! eval "$cmd"; then
            log_error "数据生成失败: $suffix"
            return 1
        fi
        log_success "数据生成完成: $suffix"
    fi
}

# 运行训练
run_training() {
    local clients="$1"
    local suffix="$2"
    local exp_name="$3"
    
    log_header "开始训练实验: $exp_name"
    
    local data_dir="$OUTPUT_DIR/data/$suffix"
    local model_output="$OUTPUT_DIR/models/$suffix"
    local log_file="$OUTPUT_DIR/logs/train_$suffix.log"
    
    # 构建客户端参数
    local client_args=""
    IFS=',' read -ra CLIENT_ARRAY <<< "$clients"
    for client in "${CLIENT_ARRAY[@]}"; do
        client_args="$client_args --enabled_clients $client"
    done
    
    local cmd="python train.py \
        --data_dir \"$data_dir\" \
        --output_dir \"$model_output\" \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --max_samples $MAX_SAMPLES \
        $client_args \
        2>&1 | tee \"$log_file\""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] $cmd"
    else
        if ! eval "$cmd"; then
            log_error "训练失败: $exp_name"
            return 1
        fi
        log_success "训练完成: $exp_name"
    fi
}

# 运行评估
run_evaluation() {
    local clients="$1"
    local suffix="$2"
    local exp_name="$3"
    
    log_header "开始评估实验: $exp_name"
    
    local data_dir="$OUTPUT_DIR/data/$suffix"
    local model_path="$OUTPUT_DIR/models/$suffix/federated_qwen_model.pth"
    local result_file="$OUTPUT_DIR/results/eval_$suffix.json"
    local log_file="$OUTPUT_DIR/logs/eval_$suffix.log"
    
    local cmd="python evaluate.py \
        --checkpoint_path \"$model_path\" \
        --test_data \"$data_dir/enhanced_qwen_test.jsonl\" \
        --max_samples 1000 \
        --output_file \"$result_file\" \
        2>&1 | tee \"$log_file\""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] $cmd"
    else
        if ! eval "$cmd"; then
            log_error "评估失败: $exp_name"
            return 1
        fi
        log_success "评估完成: $exp_name"
    fi
}

# 运行单个实验
run_single_experiment() {
    local exp_type="$1"
    
    local clients=""
    local suffix=""
    local name=""
    
    case $exp_type in
        "baseline")
            clients="port,railway,customs"
            suffix="baseline_all_clients"
            name="完整联邦学习（港口+铁路+海关）"
            ;;
        "no_port")
            clients="railway,customs"
            suffix="ablation_no_port"
            name="去除港口客户端（铁路+海关）"
            ;;
        "no_railway")
            clients="port,customs"
            suffix="ablation_no_railway"
            name="去除铁路客户端（港口+海关）"
            ;;
        "no_customs")
            clients="port,railway"
            suffix="ablation_no_customs"
            name="去除海关客户端（港口+铁路）"
            ;;
        *)
            log_error "未知实验类型: $exp_type"
            return 1
            ;;
    esac
    
    log_header "=========================================="
    log_header "实验: $name"
    log_header "客户端: $clients"
    log_header "输出后缀: $suffix"
    log_header "=========================================="
    
    # 生成数据
    generate_data "$clients" "$suffix" || return 1
    
    # 训练
    run_training "$clients" "$suffix" "$name" || return 1
    
    # 评估
    run_evaluation "$clients" "$suffix" "$name" || return 1
    
    log_success "实验完成: $name"
    echo ""
}

# 主函数
main() {
    log_header "🧪 开始消融实验..."
    echo "=========================================="
    log_info "实验配置:"
    log_info "  - 实验类型: $EXPERIMENT"
    log_info "  - 训练轮数: $EPOCHS"
    log_info "  - 学习率: $LEARNING_RATE"
    log_info "  - 批次大小: $BATCH_SIZE"
    log_info "  - 最大样本数: $MAX_SAMPLES"
    log_info "  - 输出目录: $OUTPUT_DIR"
    log_info "  - 跳过数据生成: $SKIP_DATA_GEN"
    log_info "  - 试运行模式: $DRY_RUN"
    echo "=========================================="
    
    # 创建输出目录结构
    create_output_structure "$OUTPUT_DIR"
    
    # 记录实验开始时间
    start_time=$(date)
    log_info "实验开始时间: $start_time"
    
    # 运行实验
    if [[ "$EXPERIMENT" == "all" ]]; then
        log_header "运行所有消融实验..."
        run_single_experiment "baseline" || exit 1
        run_single_experiment "no_port" || exit 1
        run_single_experiment "no_railway" || exit 1
        run_single_experiment "no_customs" || exit 1
    else
        run_single_experiment "$EXPERIMENT" || exit 1
    fi
    
    # 记录实验结束时间
    end_time=$(date)
    log_success "实验结束时间: $end_time"
    
    # 生成实验报告
    if [[ "$DRY_RUN" == "false" && "$EXPERIMENT" == "all" ]]; then
        log_header "生成消融实验报告..."
        bash "$SCRIPT_DIR/generate_ablation_report.sh" --input_dir "$OUTPUT_DIR" --output_file "$OUTPUT_DIR/reports/ablation_analysis.md"
    fi
    
    log_success "🎉 所有消融实验完成!"
    log_info "结果保存在: $OUTPUT_DIR"
}

# 检查必要文件和依赖
check_dependencies() {
    local missing_files=()
    
    if [[ ! -f "enhanced_data_processor.py" ]]; then
        missing_files+=("enhanced_data_processor.py")
    fi
    
    if [[ ! -f "train.py" ]]; then
        missing_files+=("train.py")
    fi
    
    if [[ ! -f "evaluate.py" ]]; then
        missing_files+=("evaluate.py")
    fi
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "缺少必要文件:"
        for file in "${missing_files[@]}"; do
            log_error "  - $file"
        done
        exit 1
    fi
}

# 执行主函数
check_dependencies
main

log_info "消融实验脚本执行完成。"
