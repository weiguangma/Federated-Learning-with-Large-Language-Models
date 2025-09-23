#!/usr/bin/env python3
# /********************************************************************************
#  * @Author: zhangqiuhong
#  * @Date: 2025-09-20
#  * @Description: 联邦学习Qwen模型评估脚本 - 统一评估逻辑
#  ********************************************************************************/

import torch
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import re
from collections import Counter
from federated_model import FederatedQwenSystem

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedModelEvaluator:
    """联邦模型评估器"""
    
    def __init__(self, model_path: str, checkpoint_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # 加载模型
        logger.info("🔧 加载联邦Qwen模型...")
        self.model = FederatedQwenSystem(model_path=model_path, device=device)
        
        # 加载检查点
        if Path(checkpoint_path).exists():
            logger.info(f"📥 加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 兼容两种保存格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 标准格式：包含model_state_dict键的字典
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ 模型检查点加载成功（标准格式）")
            else:
                # 简单格式：直接是state_dict
                self.model.load_state_dict(checkpoint)
                logger.info("✅ 模型检查点加载成功（简单格式）")
        else:
            logger.warning(f"⚠️  检查点文件不存在: {checkpoint_path}")
        
        self.model.to(device)
        self.model.eval()
        
        logger.info("✅ 联邦模型评估器初始化完成")
    
    def load_test_data(self, data_file: Path, max_samples: int = None) -> List[Dict]:
        """加载测试数据"""
        logger.info(f"📊 从 {data_file} 加载测试数据...")
        
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
        
        logger.info(f"✅ 成功加载 {len(samples)} 个测试样本")
        return samples
    
    def parse_qwen_sample(self, sample: dict) -> Tuple[str, Dict[str, str], str]:
        """解析标准Qwen格式的数据"""
        full_instruction = sample['instruction']
        
        # 从instruction中提取服务端和客户端指令
        parts = full_instruction.split('<|object_ref_start|>')
        server_instruction = parts[0].strip()
        
        # 提取客户端指令
        client_instructions = {}
        for i in range(1, min(4, len(parts))):  # 最多3个客户端
            client_part = parts[i]
            if '<|object_ref_end|>' in client_part:
                client_content = client_part.split('<|object_ref_end|>')[0].strip()
                client_instructions[f'client_{i:02d}'] = client_content
        
        # 确保有3个客户端指令
        for i in range(1, 4):
            key = f'client_{i:02d}'
            if key not in client_instructions:
                client_instructions[key] = "无额外信息"
        
        return server_instruction, client_instructions, sample['output']
    
    def extract_transport_mode(self, text: str) -> str:
        """从文本中提取运输方式"""
        transport_patterns = [
            r'运输方式选择[：:]\s*([^*\n]+)',
            r'选择\s*([^*\n]*运输[^*\n]*)',
            r'建议.*?([^*\n]*运输[^*\n]*)',
            r'(铁路运输|公路运输|海运|空运|水运|多式联运)'
        ]
        
        for pattern in transport_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        
        return "未知"
    
    def extract_destination(self, text: str) -> Tuple[str, str]:
        """从文本中提取目的地城市和省份"""
        # 目的地城市
        city_patterns = [
            r'目的地城市[：:]\s*([^*\n]+)',
            r'目的地[：:]?\s*([^*\n]*[市县区][^*\n]*)',
            r'到达\s*([^*\n]*[市县区][^*\n]*)'
        ]
        
        city = "未知"
        for pattern in city_patterns:
            match = re.search(pattern, text)
            if match:
                city = match.group(1).strip()
                break
        
        # 目的地省份
        province_patterns = [
            r'目的地省份[：:]\s*([^*\n]+)',
            r'([^*\n]*[省市区][^*\n]*)',
        ]
        
        province = "未知"
        for pattern in province_patterns:
            match = re.search(pattern, text)
            if match:
                province = match.group(1).strip()
                break
        
        return city, province
    
    def calculate_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """计算评估指标"""
        if len(predictions) != len(targets):
            logger.error("预测结果和目标结果数量不匹配")
            return {}
        
        # 提取结构化信息
        pred_transports = [self.extract_transport_mode(pred) for pred in predictions]
        target_transports = [self.extract_transport_mode(target) for target in targets]
        
        pred_cities = []
        pred_provinces = []
        target_cities = []
        target_provinces = []
        
        for pred, target in zip(predictions, targets):
            pred_city, pred_province = self.extract_destination(pred)
            target_city, target_province = self.extract_destination(target)
            
            pred_cities.append(pred_city)
            pred_provinces.append(pred_province)
            target_cities.append(target_city)
            target_provinces.append(target_province)
        
        # 计算准确率
        transport_acc = sum(1 for p, t in zip(pred_transports, target_transports) if p == t) / len(predictions)
        city_acc = sum(1 for p, t in zip(pred_cities, target_cities) if p == t) / len(predictions)
        province_acc = sum(1 for p, t in zip(pred_provinces, target_provinces) if p == t) / len(predictions)
        
        # 计算BLEU分数（简化版）
        def simple_bleu(pred: str, target: str) -> float:
            pred_words = set(pred.split())
            target_words = set(target.split())
            if len(target_words) == 0:
                return 0.0
            intersection = pred_words.intersection(target_words)
            return len(intersection) / len(target_words)
        
        bleu_scores = [simple_bleu(pred, target) for pred, target in zip(predictions, targets)]
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        
        # 计算完整匹配准确率
        exact_match = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip()) / len(predictions)
        
        metrics = {
            'transport_accuracy': transport_acc,
            'city_accuracy': city_acc,
            'province_accuracy': province_acc,
            'avg_bleu_score': avg_bleu,
            'exact_match_accuracy': exact_match,
            'total_samples': len(predictions)
        }
        
        return metrics
    
    def evaluate_model(self, test_data: List[Dict], max_new_tokens: int = 200, 
                      temperature: float = 0.7, batch_size: int = 1) -> Dict:
        """评估模型性能"""
        logger.info("🧪 开始模型评估...")
        
        predictions = []
        targets = []
        generation_errors = 0
        
        with torch.no_grad():
            for i, sample in enumerate(tqdm(test_data, desc="评估进度")):
                try:
                    # 解析样本
                    server_instruction, client_instructions, target_output = self.parse_qwen_sample(sample)
                    
                    # 生成预测
                    generated_text = self.model.generate(
                        server_instruction,
                        client_instructions,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True
                    )
                    
                    predictions.append(generated_text)
                    targets.append(target_output)
                    
                except Exception as e:
                    logger.warning(f"样本 {i} 生成失败: {e}")
                    generation_errors += 1
                    predictions.append("生成失败")
                    targets.append(sample.get('output', ''))
        
        # 计算指标
        logger.info("📊 计算评估指标...")
        metrics = self.calculate_metrics(predictions, targets)
        metrics['generation_errors'] = generation_errors
        metrics['generation_success_rate'] = 1.0 - (generation_errors / len(test_data))
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets
        }
    
    def print_evaluation_report(self, results: Dict):
        """打印评估报告"""
        metrics = results['metrics']
        predictions = results['predictions']
        targets = results['targets']
        
        logger.info("=" * 80)
        logger.info("📈 联邦Qwen模型评估报告")
        logger.info("=" * 80)
        
        logger.info(f"📊 基础指标:")
        logger.info(f"   - 总样本数: {metrics['total_samples']}")
        logger.info(f"   - 生成成功率: {metrics['generation_success_rate']:.2%}")
        logger.info(f"   - 生成错误数: {metrics['generation_errors']}")
        
        logger.info(f"🎯 任务准确率:")
        logger.info(f"   - 运输方式准确率: {metrics['transport_accuracy']:.2%}")
        logger.info(f"   - 目的地城市准确率: {metrics['city_accuracy']:.2%}")
        logger.info(f"   - 目的地省份准确率: {metrics['province_accuracy']:.2%}")
        
        logger.info(f"📝 文本质量指标:")
        logger.info(f"   - 平均BLEU分数: {metrics['avg_bleu_score']:.4f}")
        logger.info(f"   - 完全匹配准确率: {metrics['exact_match_accuracy']:.2%}")
        
        # 显示一些示例
        logger.info(f"🔍 生成示例 (前3个):")
        for i in range(min(3, len(predictions))):
            logger.info(f"   样本 {i+1}:")
            logger.info(f"     目标: {targets[i][:100]}...")
            logger.info(f"     预测: {predictions[i][:100]}...")
            logger.info(f"     ---")
        
        logger.info("=" * 80)
    
    def save_evaluation_results(self, results: Dict, output_file: str):
        """保存评估结果"""
        logger.info(f"💾 保存评估结果到: {output_file}")
        
        # 准备保存数据
        save_data = {
            'metrics': results['metrics'],
            'model_path': self.model_path,
            'checkpoint_path': self.checkpoint_path,
            'evaluation_samples': []
        }
        
        # 添加样本详情（限制数量以避免文件过大）
        max_save_samples = min(100, len(results['predictions']))
        for i in range(max_save_samples):
            save_data['evaluation_samples'].append({
                'sample_id': i,
                'prediction': results['predictions'][i],
                'target': results['targets'][i]
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info("✅ 评估结果保存完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="评估联邦Qwen模型")
    parser.add_argument("--model_path", type=str,
                       default="/root/autodl-tmp/Federated_learning/llm_init_ckpt/Qwen2.5-0.5B-Instruct",
                       help="预训练模型路径")
    parser.add_argument("--checkpoint_path", type=str,
                       default="federated_qwen_output/federated_qwen_model.pth",
                       help="训练后的检查点路径")
    parser.add_argument("--test_data", type=str,
                       default="./data/qwen_processed/qwen_federated_train.jsonl",
                       help="测试数据文件")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="最大测试样本数")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                       help="生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="生成温度")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="评估结果输出文件")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    
    args = parser.parse_args()
    
    try:
        # 初始化评估器
        evaluator = FederatedModelEvaluator(
            model_path=args.model_path,
            checkpoint_path=args.checkpoint_path,
            device=args.device
        )
        
        # 加载测试数据
        test_data_file = Path(args.test_data)
        if not test_data_file.exists():
            logger.error(f"测试数据文件不存在: {test_data_file}")
            return False
        
        test_data = evaluator.load_test_data(test_data_file, args.max_samples)
        
        if len(test_data) == 0:
            logger.error("没有找到有效的测试数据")
            return False
        
        # 执行评估
        results = evaluator.evaluate_model(
            test_data=test_data,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        # 打印报告
        evaluator.print_evaluation_report(results)
        
        # 保存结果
        evaluator.save_evaluation_results(results, args.output_file)
        
        logger.info("🎉 模型评估完成！")
        return True
        
    except Exception as e:
        logger.error(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)