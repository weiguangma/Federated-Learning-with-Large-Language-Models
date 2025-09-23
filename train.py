#!/usr/bin/env python3
# /********************************************************************************
#  * @Author: zhangqiuhong
#  * @Date: 2025-09-20
#  * @Description: 联邦学习Qwen模型训练脚本 - 清理版本
#  ********************************************************************************/

import torch
import torch.nn as nn
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import random
import re
from typing import List, Dict, Tuple, Optional
from federated_model import FederatedQwenSystem

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data(data_file: Path, max_samples: int = None):
    """加载训练数据"""
    logger.info(f"📊 从 {data_file} 加载训练数据...")
    
    samples = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if max_samples and len(samples) >= max_samples:
                        break
                
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"跳过无效JSON行 {line_idx}: {e}")
                continue
    
    logger.info(f"✅ 成功加载 {len(samples)} 个训练样本")
    return samples

def parse_qwen_sample(sample: dict):
    """解析Qwen格式的数据 - 支持增强版和旧版格式"""
    
    # 检查是否是增强版格式（直接包含 port_expert, railway_expert, customs_expert）
    if 'port_expert' in sample and 'railway_expert' in sample and 'customs_expert' in sample:
        # 增强版格式
        server_instruction = sample.get('server_instruction', "请根据以下信息做出最优运输决策")
        client_instructions = {
            'port': sample['port_expert'],
            'railway': sample['railway_expert'], 
            'customs': sample['customs_expert']
        }
        return server_instruction, client_instructions, sample['output']
    
    # 旧版格式处理
    full_instruction = sample['instruction']
    
    # 从instruction中提取服务端和客户端指令
    parts = full_instruction.split('<|object_ref_start|>')
    server_instruction = parts[0].strip()
    
    # 提取客户端指令
    client_instructions = {}
    client_names = ['port', 'railway', 'customs']  # 使用正确的客户端名称
    
    for i in range(1, min(4, len(parts))):  # 最多3个客户端
        client_part = parts[i]
        if '<|object_ref_end|>' in client_part:
            client_content = client_part.split('<|object_ref_end|>')[0].strip()
            client_name = client_names[i-1] if i-1 < len(client_names) else f'client_{i}'
            client_instructions[client_name] = client_content
    
    # 确保有3个客户端指令
    for client_name in client_names:
        if client_name not in client_instructions:
            client_instructions[client_name] = "无额外信息"
    
    return server_instruction, client_instructions, sample['output']

def evaluate_test_accuracy(federated_model, test_samples: List[Dict], max_samples: int = 50) -> Dict:
    """在训练过程中快速评估测试集准确率"""
    if not test_samples:
        return {'transport_accuracy': 0, 'city_accuracy': 0, 'province_accuracy': 0, 'overall_accuracy': 0, 'total_samples': 0}
    
    eval_samples = random.sample(test_samples, min(max_samples, len(test_samples)))
    
    correct_transport = 0
    correct_city = 0  
    correct_province = 0
    total_samples = len(eval_samples)
    
    federated_model.eval()
    with torch.no_grad():
        for sample in eval_samples:
            try:
                server_instruction, client_instructions, expected_output = parse_qwen_sample(sample)
                
                # 生成预测结果
                generated_text = federated_model.generate(
                    server_instruction=server_instruction,
                    client_instructions=client_instructions,
                    max_new_tokens=100,
                    temperature=0.1
                )
                
                # 简单的准确率评估
                if "铁路运输" in expected_output and "铁路运输" in generated_text:
                    correct_transport += 1
                elif "公路运输" in expected_output and "公路运输" in generated_text:
                    correct_transport += 1
                    
            except Exception as e:
                logger.debug(f"评估样本失败: {e}")
                continue
    
    federated_model.train()
    
    return {
        'transport_accuracy': correct_transport / total_samples if total_samples > 0 else 0,
        'city_accuracy': 0,  # 简化版本
        'province_accuracy': 0,  # 简化版本
        'overall_accuracy': correct_transport / total_samples if total_samples > 0 else 0,
        'total_samples': total_samples
    }

def train_federated_qwen(
    data_dir: str = "./data",
    model_path: str = "/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct",
    epochs: int = 3,
    learning_rate: float = 2e-6,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    max_length: int = 512,
    save_steps: int = 500,
    output_dir: str = "federated_qwen_output",
    max_samples: int = None,
    eval_steps: int = 1000
):
    """训练联邦Qwen模型"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"📱 使用设备: {device}")
    
    try:
        # 1. 加载联邦模型
        logger.info("🔧 初始化联邦Qwen模型...")
        federated_model = FederatedQwenSystem(model_path=model_path, device=device)
        federated_model.to(device)
        federated_model.train()
        
        logger.info("✅ 模型初始化完成")
        
        # 2. 加载数据
        logger.info("📊 加载训练数据...")
        # 支持新的增强版数据和旧版数据
        if (Path(data_dir) / "enhanced_qwen_train.jsonl").exists():
            train_file = Path(data_dir) / "enhanced_qwen_train.jsonl"
            test_file = Path(data_dir) / "enhanced_qwen_test.jsonl"
        elif (Path(data_dir) / "qwen_processed" / "qwen_federated_train.jsonl").exists():
            train_file = Path(data_dir) / "qwen_processed" / "qwen_federated_train.jsonl"
            test_file = Path(data_dir) / "qwen_processed" / "qwen_federated_test.jsonl"
        else:
            raise FileNotFoundError(f"在 {data_dir} 中找不到训练数据文件")
        
        if not train_file.exists():
            raise FileNotFoundError(f"训练数据文件不存在: {train_file}")
            
        training_samples = load_training_data(train_file, max_samples)
        test_samples = []
        if test_file.exists():
            test_samples = load_training_data(test_file, 200)
        
        if len(training_samples) == 0:
            raise ValueError("没有找到有效的训练样本")
        
        # 3. 设置优化器
        optimizer = torch.optim.AdamW(
            federated_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        max_grad_norm = 1.0
        
        logger.info(f"🚀 开始训练...")
        logger.info(f"   - 训练样本数: {len(training_samples)}")
        logger.info(f"   - 训练轮数: {epochs}")
        logger.info(f"   - 学习率: {learning_rate}")
        logger.info(f"   - 批次大小: {batch_size}")
        logger.info(f"   - 梯度累积步数: {gradient_accumulation_steps}")
        
        # 4. 训练循环
        total_loss = 0.0
        valid_steps = 0
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_valid_steps = 0
            
            progress_bar = tqdm(training_samples, desc=f"Epoch {epoch+1}")
            
            for step, sample in enumerate(progress_bar):
                try:
                    # 解析样本
                    server_instruction, client_instructions, expected_output = parse_qwen_sample(sample)
                    
                    # 前向传播
                    outputs = federated_model(
                        server_instruction=server_instruction,
                        client_instructions=client_instructions,
                        target_output=expected_output
                    )
                    
                    # 检查损失
                    if "loss" in outputs:
                        loss = outputs["loss"]
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            logger.warning(f"跳过无效损失: {loss.item()}")
                            continue
                        
                        # 梯度累积
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        
                        # 检查是否更新参数
                        if (step + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                            torch.nn.utils.clip_grad_norm_(federated_model.parameters(), max_grad_norm)
                
                            # 更新参数
                            optimizer.step()
                
                            # 联邦学习同步
                            federated_model.federated_step()
                            
                            optimizer.zero_grad()
                            
                            # 定期评估
                            if test_samples and global_step > 0 and global_step % eval_steps == 0:
                                logger.info(f"\n📊 步骤 {global_step}: 开始测试集评估...")
                                eval_results = evaluate_test_accuracy(federated_model, test_samples, max_samples=50)
                                logger.info(f"📈 测试集准确率: {eval_results['overall_accuracy']:.2%}")
                            
                            global_step += 1
                        
                        # 记录损失
                        loss_item = loss.item()
                        total_loss += loss_item
                        epoch_loss += loss_item
                        valid_steps += 1
                        epoch_valid_steps += 1
                        
                        # 更新进度条
                        avg_loss = total_loss / valid_steps
                        progress_bar.set_postfix({
                            'loss': f'{loss_item:.4f}',
                            'avg_loss': f'{avg_loss:.4f}',
                            'valid_steps': valid_steps
                        })
                
                except Exception as e:
                    logger.warning(f"处理样本失败: {e}")
                    continue
            
            # Epoch结束
            if epoch_valid_steps > 0:
                avg_epoch_loss = epoch_loss / epoch_valid_steps
                logger.info(f"✅ Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        # 5. 保存模型
        logger.info("💾 保存训练后的模型...")
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存模型（标准格式）
        checkpoint = {
            'model_state_dict': federated_model.state_dict(),
            'epoch': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'total_loss': total_loss,
            'valid_steps': valid_steps,
            'avg_loss': total_loss / valid_steps if valid_steps > 0 else 0
        }
        torch.save(checkpoint, output_path / "federated_qwen_model.pth")
        logger.info(f"模型检查点已保存到: {output_path / 'federated_qwen_model.pth'}")
        
        # 保存训练配置
        config = {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'total_loss': total_loss,
            'valid_steps': valid_steps,
            'avg_loss': total_loss / valid_steps if valid_steps > 0 else 0
        }
        
        with open(output_path / "training_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 训练完成！模型已保存到: {output_dir}")
        logger.info(f"   - 有效训练步骤: {valid_steps}")
        logger.info(f"   - 最终平均损失: {total_loss / valid_steps if valid_steps > 0 else 0:.4f}")
    
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练联邦Qwen模型")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--model_path", type=str, 
                       default="/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct",
                       help="模型路径")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="学习率")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=1000, help="评估步数")
    parser.add_argument("--output_dir", type=str, default="federated_qwen_output", help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    
    args = parser.parse_args()
    
    # 开始训练
    success = train_federated_qwen(
        data_dir=args.data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        eval_steps=args.eval_steps
    )
    
    if success:
        logger.info("🎉 训练成功完成！")
    else:
        logger.error("❌ 训练失败！")
        exit(1)

if __name__ == "__main__":
    main()
