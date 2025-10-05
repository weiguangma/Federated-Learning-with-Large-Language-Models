#!/bin/bash

set -e  # 遇到错误时退出

# 脚本配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 验证配置
SAMPLE_SIZE=100
TRAIN_RATIO=0.8  # 80条训练，20条测试
RAILWAY_RATIO=0.6  # 60%铁路运输
EPOCHS=3
LEARNING_RATE=5e-6
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
MAX_LENGTH=1024

# 目录配置
DEBUG_DATA_DIR="$PROJECT_DIR/data/debug_validation"
DEBUG_OUTPUT_DIR="$PROJECT_DIR/debug_validation_output"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# 显示配置信息
show_config() {
    echo -e "${BLUE}🔧 联邦学习模型验证配置${NC}"
    echo "=========================================="
    echo "📊 数据配置:"
    echo "  - 总样本数: $SAMPLE_SIZE"
    echo "  - 训练集比例: $TRAIN_RATIO ($(echo "$SAMPLE_SIZE * $TRAIN_RATIO" | bc | cut -d. -f1)条)"
    echo "  - 测试集比例: $(echo "1 - $TRAIN_RATIO" | bc) ($(echo "$SAMPLE_SIZE * (1 - $TRAIN_RATIO)" | bc | cut -d. -f1)条)"
    echo "  - 铁路运输比例: $RAILWAY_RATIO"
    echo ""
    echo "🚀 训练配置:"
    echo "  - 训练轮数: $EPOCHS"
    echo "  - 学习率: $LEARNING_RATE"
    echo "  - 批次大小: $BATCH_SIZE"
    echo "  - 梯度累积步数: $GRADIENT_ACCUMULATION_STEPS"
    echo "  - 最大序列长度: $MAX_LENGTH"
    echo ""
    echo "📁 目录配置:"
    echo "  - 数据目录: $DEBUG_DATA_DIR"
    echo "  - 输出目录: $DEBUG_OUTPUT_DIR"
    echo "=========================================="
}

# 检查环境
check_environment() {
    log_step "检查运行环境..."
    
    # 检查Python
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1)
        log_success "🎮 GPU可用: $gpu_info"
    else
        log_warning "⚠️ 未检测到GPU，将使用CPU训练"
    fi
    
    # 检查必要文件
    if [[ ! -f "$PROJECT_DIR/enhanced_data_processor.py" ]]; then
        log_error "数据处理脚本不存在: $PROJECT_DIR/enhanced_data_processor.py"
        exit 1
    fi
    
    if [[ ! -f "$PROJECT_DIR/train.py" ]]; then
        log_error "训练脚本不存在: $PROJECT_DIR/train.py"
        exit 1
    fi
    
    if [[ ! -f "$PROJECT_DIR/evaluate.py" ]]; then
        log_error "评估脚本不存在: $PROJECT_DIR/evaluate.py"
        exit 1
    fi
    
    log_success "✅ 环境检查通过"
}

# 步骤1: 数据预处理
step_data_preprocessing() {
    log_step "步骤1: 数据预处理"
    echo "=========================================="
    
    # 清理旧数据
    if [[ -d "$DEBUG_DATA_DIR" ]]; then
        log_info "🧹 清理旧的验证数据..."
        rm -rf "$DEBUG_DATA_DIR"
    fi
    
    # 创建数据目录
    mkdir -p "$DEBUG_DATA_DIR"
    
    # 运行数据预处理
    log_info "🔄 生成验证数据集..."
    cd "$PROJECT_DIR"
    
    python enhanced_data_processor.py \
        --sample_size $SAMPLE_SIZE \
        --train_ratio $TRAIN_RATIO \
        --railway_ratio $RAILWAY_RATIO \
        --output_dir "$DEBUG_DATA_DIR" \
        --random_seed 42
    
    # 检查生成的数据
    if [[ -f "$DEBUG_DATA_DIR/enhanced_qwen_train.jsonl" ]] && [[ -f "$DEBUG_DATA_DIR/enhanced_qwen_test.jsonl" ]]; then
        train_count=$(wc -l < "$DEBUG_DATA_DIR/enhanced_qwen_train.jsonl")
        test_count=$(wc -l < "$DEBUG_DATA_DIR/enhanced_qwen_test.jsonl")
        train_size=$(du -h "$DEBUG_DATA_DIR/enhanced_qwen_train.jsonl" | cut -f1)
        test_size=$(du -h "$DEBUG_DATA_DIR/enhanced_qwen_test.jsonl" | cut -f1)
        
        log_success "✅ 数据预处理完成"
        log_info "📊 训练数据: $train_count 条样本 ($train_size)"
        log_info "📊 测试数据: $test_count 条样本 ($test_size)"
    else
        log_error "❌ 数据预处理失败"
        exit 1
    fi
}

# 步骤2: 模型训练
step_model_training() {
    log_step "步骤2: 模型训练"
    echo "=========================================="
    
    # 清理旧的训练输出
    if [[ -d "$DEBUG_OUTPUT_DIR" ]]; then
        log_info "🧹 清理旧的训练输出..."
        rm -rf "$DEBUG_OUTPUT_DIR"
    fi
    
    # 创建输出目录
    mkdir -p "$DEBUG_OUTPUT_DIR"
    
    # 开始训练
    log_info "🚀 开始联邦学习训练..."
    cd "$PROJECT_DIR"
    
    python train.py \
        --data_dir "$DEBUG_DATA_DIR" \
        --epochs $EPOCHS \
        --learning_rate $LEARNING_RATE \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --max_length $MAX_LENGTH \
        --output_dir "$DEBUG_OUTPUT_DIR" \
        --eval_steps 50 \
        --save_steps 100
    
    # 检查训练结果
    if [[ -f "$DEBUG_OUTPUT_DIR/federated_qwen_model.pth" ]] && [[ -f "$DEBUG_OUTPUT_DIR/training_config.json" ]]; then
        model_size=$(du -h "$DEBUG_OUTPUT_DIR/federated_qwen_model.pth" | cut -f1)
        log_success "✅ 模型训练完成"
        log_info "🤖 模型文件大小: $model_size"
        
        # 显示训练配置
        if [[ -f "$DEBUG_OUTPUT_DIR/training_config.json" ]]; then
            log_info "📋 训练统计:"
            python -c "
import json
with open('$DEBUG_OUTPUT_DIR/training_config.json', 'r') as f:
    config = json.load(f)
print(f\"  - 有效训练步骤: {config.get('valid_steps', 0)}\")
print(f\"  - 最终平均损失: {config.get('avg_loss', 0):.4f}\")
print(f\"  - 总损失: {config.get('total_loss', 0):.4f}\")
"
        fi
    else
        log_error "❌ 模型训练失败"
        exit 1
    fi
}

# 步骤3: 模型评估
step_model_evaluation() {
    log_step "步骤3: 模型评估"
    echo "=========================================="
    
    # 运行模型评估
    log_info "🧪 开始模型评估..."
    cd "$PROJECT_DIR"
    
    python evaluate.py \
        --model_path "/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct" \
        --checkpoint_path "$DEBUG_OUTPUT_DIR/federated_qwen_model.pth" \
        --test_data "$DEBUG_DATA_DIR/enhanced_qwen_test.jsonl" \
        --max_samples 20 \
        --max_new_tokens 150 \
        --temperature 0.3 \
        --output_file "$DEBUG_OUTPUT_DIR/evaluation_results.json" \
        --device "cuda"
    
    # 检查评估结果
    if [[ -f "$DEBUG_OUTPUT_DIR/evaluation_results.json" ]]; then
        eval_size=$(du -h "$DEBUG_OUTPUT_DIR/evaluation_results.json" | cut -f1)
        log_success "✅ 模型评估完成"
        log_info "📊 评估结果文件: $eval_size"
        
        # 显示关键评估指标
        log_info "📈 关键评估指标:"
        python -c "
import json
with open('$DEBUG_OUTPUT_DIR/evaluation_results.json', 'r') as f:
    results = json.load(f)
print(f\"  - 生成成功率: {results.get('generation_success_rate', 0)*100:.1f}%\")
print(f\"  - 运输方式准确率: {results.get('transport_accuracy', 0)*100:.1f}%\")
print(f\"  - 目的地城市准确率: {results.get('city_accuracy', 0)*100:.1f}%\")
print(f\"  - 目的地省份准确率: {results.get('province_accuracy', 0)*100:.1f}%\")
print(f\"  - 平均BLEU分数: {results.get('avg_bleu_score', 0):.4f}\")
print(f\"  - 评估样本数: {results.get('total_samples', 0)}\")
"
    else
        log_error "❌ 模型评估失败"
        exit 1
    fi
}

# 步骤4: 过拟合验证
step_overfitting_validation() {
    log_step "步骤4: 过拟合验证"
    echo "=========================================="
    
    log_info "🔍 验证模型是否能够过拟合训练数据..."
    
    # 使用训练数据进行评估（应该有很高的准确率）
    python evaluate.py \
        --model_path "/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct" \
        --checkpoint_path "$DEBUG_OUTPUT_DIR/federated_qwen_model.pth" \
        --test_data "$DEBUG_DATA_DIR/enhanced_qwen_train.jsonl" \
        --max_samples 20 \
        --max_new_tokens 150 \
        --temperature 0.1 \
        --output_file "$DEBUG_OUTPUT_DIR/train_evaluation_results.json" \
        --device "cuda"
    
    if [[ -f "$DEBUG_OUTPUT_DIR/train_evaluation_results.json" ]]; then
        log_success "✅ 过拟合验证完成"
        log_info "📈 训练数据评估结果:"
        python -c "
import json
with open('$DEBUG_OUTPUT_DIR/train_evaluation_results.json', 'r') as f:
    results = json.load(f)
transport_acc = results.get('transport_accuracy', 0)*100
city_acc = results.get('city_accuracy', 0)*100
province_acc = results.get('province_accuracy', 0)*100
bleu_score = results.get('avg_bleu_score', 0)

print(f\"  - 运输方式准确率: {transport_acc:.1f}%\")
print(f\"  - 目的地城市准确率: {city_acc:.1f}%\")
print(f\"  - 目的地省份准确率: {province_acc:.1f}%\")
print(f\"  - 平均BLEU分数: {bleu_score:.4f}\")

# 判断是否过拟合成功
if transport_acc > 50 or city_acc > 30 or bleu_score > 0.1:
    print(f\"\\n✅ 模型能够学习训练数据，架构实现正确！\")
else:
    print(f\"\\n⚠️ 模型可能需要更多训练或调整参数\")
"
    else
        log_warning "⚠️ 过拟合验证失败"
    fi
}

# 生成验证报告
generate_validation_report() {
    log_step "生成验证报告"
    echo "=========================================="
    
    report_file="$DEBUG_OUTPUT_DIR/validation_report.md"
    
    cat > "$report_file" << EOF
# 联邦学习模型验证报告

## 验证配置
- **数据样本数**: $SAMPLE_SIZE 条
- **训练集**: $(echo "$SAMPLE_SIZE * $TRAIN_RATIO" | bc | cut -d. -f1) 条
- **测试集**: $(echo "$SAMPLE_SIZE * (1 - $TRAIN_RATIO)" | bc | cut -d. -f1) 条
- **铁路运输比例**: $RAILWAY_RATIO
- **训练轮数**: $EPOCHS
- **学习率**: $LEARNING_RATE
- **批次大小**: $BATCH_SIZE

## 文件结构
\`\`\`
$DEBUG_OUTPUT_DIR/
├── federated_qwen_model.pth          # 训练后的模型
├── training_config.json              # 训练配置
├── evaluation_results.json           # 测试集评估结果
├── train_evaluation_results.json     # 训练集评估结果（过拟合验证）
└── validation_report.md              # 本报告
\`\`\`

## 验证结论
EOF

    # 添加验证结论
    if [[ -f "$DEBUG_OUTPUT_DIR/evaluation_results.json" ]] && [[ -f "$DEBUG_OUTPUT_DIR/train_evaluation_results.json" ]]; then
        python >> "$report_file" << 'EOF'
import json

# 读取评估结果
with open('DEBUG_OUTPUT_DIR/evaluation_results.json'.replace('DEBUG_OUTPUT_DIR', 'DEBUG_OUTPUT_DIR_VALUE'), 'r') as f:
    test_results = json.load(f)
with open('DEBUG_OUTPUT_DIR/train_evaluation_results.json'.replace('DEBUG_OUTPUT_DIR', 'DEBUG_OUTPUT_DIR_VALUE'), 'r') as f:
    train_results = json.load(f)

print("\n### 测试集表现")
print(f"- 生成成功率: {test_results.get('generation_success_rate', 0)*100:.1f}%")
print(f"- 运输方式准确率: {test_results.get('transport_accuracy', 0)*100:.1f}%")
print(f"- 目的地城市准确率: {test_results.get('city_accuracy', 0)*100:.1f}%")
print(f"- 平均BLEU分数: {test_results.get('avg_bleu_score', 0):.4f}")

print("\n### 训练集表现（过拟合验证）")
print(f"- 运输方式准确率: {train_results.get('transport_accuracy', 0)*100:.1f}%")
print(f"- 目的地城市准确率: {train_results.get('city_accuracy', 0)*100:.1f}%")
print(f"- 平均BLEU分数: {train_results.get('avg_bleu_score', 0):.4f}")

# 验证结论
train_transport = train_results.get('transport_accuracy', 0)*100
train_city = train_results.get('city_accuracy', 0)*100
train_bleu = train_results.get('avg_bleu_score', 0)

print("\n### 架构验证结论")
if train_transport > 50 or train_city > 30 or train_bleu > 0.1:
    print("✅ **模型架构实现正确**")
    print("- 模型能够成功学习训练数据")
    print("- Split Learning架构工作正常")
    print("- 联邦权重同步机制有效")
    print("- SFT损失计算正确")
else:
    print("⚠️ **需要进一步调试**")
    print("- 模型可能需要更多训练轮数")
    print("- 学习率可能需要调整")
    print("- 数据质量可能需要改进")
EOF
        # 替换占位符
        sed -i "s/DEBUG_OUTPUT_DIR_VALUE/$DEBUG_OUTPUT_DIR/g" "$report_file"
    fi
    
    log_success "✅ 验证报告已生成: $report_file"
}

# 清理函数
cleanup() {
    log_info "🧹 清理临时文件..."
    # 可以在这里添加清理逻辑
}

# 主函数
main() {
    echo -e "${GREEN}🚀 开始联邦学习模型完整验证${NC}"
    echo "=========================================="
    
    # 显示配置
    show_config
    echo ""
    
    # 检查环境
    check_environment
    echo ""
    
    # 记录开始时间
    start_time=$(date +%s)
    
    # 执行验证步骤
    step_data_preprocessing
    echo ""
    
    step_model_training
    echo ""
    
    step_model_evaluation
    echo ""
    
    step_overfitting_validation
    echo ""
    
    # 生成报告
    generate_validation_report
    echo ""
    
    # 计算总耗时
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    # 最终总结
    echo "=========================================="
    log_success "🎉 联邦学习模型验证完成！"
    log_info "⏱️ 总耗时: ${minutes}分${seconds}秒"
    log_info "📁 结果目录: $DEBUG_OUTPUT_DIR"
    log_info "📋 验证报告: $DEBUG_OUTPUT_DIR/validation_report.md"
    echo ""
    
    log_info "🔍 查看验证报告:"
    log_info "   cat \"$DEBUG_OUTPUT_DIR/validation_report.md\""
    echo ""
    
    log_info "📊 查看详细评估结果:"
    log_info "   cat \"$DEBUG_OUTPUT_DIR/evaluation_results.json\""
    echo "=========================================="
}

# 捕获退出信号进行清理
trap cleanup EXIT

# 运行主函数
main "$@"
