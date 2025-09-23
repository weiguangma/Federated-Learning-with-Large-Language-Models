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
from collections import Counter, defaultdict
import numpy as np
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
    
    def calculate_classification_metrics(self, predictions: List[str], targets: List[str], task_name: str) -> Dict[str, float]:
        """计算分类任务的详细指标（精确率、召回率、F1值等）"""
        if len(predictions) != len(targets):
            return {}
        
        # 获取所有唯一类别
        all_labels = list(set(predictions + targets))
        if "未知" in all_labels:
            all_labels.remove("未知")  # 移除未知类别
        
        # 计算每个类别的TP, FP, FN
        per_class_metrics = {}
        for label in all_labels:
            tp = sum(1 for p, t in zip(predictions, targets) if p == label and t == label)
            fp = sum(1 for p, t in zip(predictions, targets) if p == label and t != label)
            fn = sum(1 for p, t in zip(predictions, targets) if p != label and t == label)
            
            # 计算精确率、召回率、F1值
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(1 for t in targets if t == label)
            }
        
        # 计算macro平均
        if all_labels:
            macro_precision = np.mean([metrics['precision'] for metrics in per_class_metrics.values()])
            macro_recall = np.mean([metrics['recall'] for metrics in per_class_metrics.values()])
            macro_f1 = np.mean([metrics['f1'] for metrics in per_class_metrics.values()])
        else:
            macro_precision = macro_recall = macro_f1 = 0.0
        
        # 计算weighted平均
        total_support = sum(metrics['support'] for metrics in per_class_metrics.values())
        if total_support > 0:
            weighted_precision = sum(metrics['precision'] * metrics['support'] for metrics in per_class_metrics.values()) / total_support
            weighted_recall = sum(metrics['recall'] * metrics['support'] for metrics in per_class_metrics.values()) / total_support
            weighted_f1 = sum(metrics['f1'] * metrics['support'] for metrics in per_class_metrics.values()) / total_support
        else:
            weighted_precision = weighted_recall = weighted_f1 = 0.0
        
        # 计算整体准确率
        accuracy = sum(1 for p, t in zip(predictions, targets) if p == t) / len(predictions)
        
        return {
            f'{task_name}_accuracy': accuracy,
            f'{task_name}_macro_precision': macro_precision,
            f'{task_name}_macro_recall': macro_recall,
            f'{task_name}_macro_f1': macro_f1,
            f'{task_name}_weighted_precision': weighted_precision,
            f'{task_name}_weighted_recall': weighted_recall,
            f'{task_name}_weighted_f1': weighted_f1,
            f'{task_name}_per_class': per_class_metrics,
            f'{task_name}_label_distribution': Counter(targets)
        }
    
    def calculate_confusion_matrix(self, predictions: List[str], targets: List[str]) -> Dict[str, Dict[str, int]]:
        """计算混淆矩阵"""
        all_labels = sorted(list(set(predictions + targets)))
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for pred, target in zip(predictions, targets):
            confusion_matrix[target][pred] += 1
        
        return dict(confusion_matrix)
    
    def calculate_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """计算评估指标 - 增强版本"""
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
        
        # 计算各任务的详细分类指标
        transport_metrics = self.calculate_classification_metrics(pred_transports, target_transports, 'transport')
        city_metrics = self.calculate_classification_metrics(pred_cities, target_cities, 'city')
        province_metrics = self.calculate_classification_metrics(pred_provinces, target_provinces, 'province')
        
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
        
        # 计算混淆矩阵
        transport_confusion = self.calculate_confusion_matrix(pred_transports, target_transports)
        city_confusion = self.calculate_confusion_matrix(pred_cities, target_cities)
        province_confusion = self.calculate_confusion_matrix(pred_provinces, target_provinces)
        
        # 合并所有指标
        all_metrics = {
            'total_samples': len(predictions),
            'avg_bleu_score': avg_bleu,
            'exact_match_accuracy': exact_match,
        }
        
        # 添加各任务的指标
        all_metrics.update(transport_metrics)
        all_metrics.update(city_metrics)
        all_metrics.update(province_metrics)
        
        # 添加混淆矩阵
        all_metrics['confusion_matrices'] = {
            'transport': transport_confusion,
            'city': city_confusion,
            'province': province_confusion
        }
        
        # 保持向后兼容性的简化指标
        all_metrics['transport_accuracy'] = transport_metrics.get('transport_accuracy', 0.0)
        all_metrics['city_accuracy'] = city_metrics.get('city_accuracy', 0.0)
        all_metrics['province_accuracy'] = province_metrics.get('province_accuracy', 0.0)
        
        return all_metrics
    
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
        """打印详细评估报告"""
        metrics = results['metrics']
        predictions = results['predictions']
        targets = results['targets']
        
        logger.info("=" * 80)
        logger.info("📈 联邦Qwen模型详细评估报告")
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
        
        # 详细分类指标报告
        self._print_task_metrics(metrics, 'transport', '🚛 运输方式分类指标')
        self._print_task_metrics(metrics, 'city', '🏙️ 目的地城市预测指标')
        self._print_task_metrics(metrics, 'province', '🗺️ 目的地省份预测指标')
        
        # 混淆矩阵展示
        self._print_confusion_matrices(metrics)
        
        # 显示一些示例
        logger.info(f"🔍 生成示例 (前3个):")
        for i in range(min(3, len(predictions))):
            logger.info(f"   样本 {i+1}:")
            logger.info(f"     目标: {targets[i][:100]}...")
            logger.info(f"     预测: {predictions[i][:100]}...")
            logger.info(f"     ---")
        
        logger.info("=" * 80)
    
    def _print_task_metrics(self, metrics: Dict, task_name: str, title: str):
        """打印单个任务的详细指标"""
        logger.info(f"\n{title}:")
        
        # 宏平均指标
        macro_precision = metrics.get(f'{task_name}_macro_precision', 0.0)
        macro_recall = metrics.get(f'{task_name}_macro_recall', 0.0)
        macro_f1 = metrics.get(f'{task_name}_macro_f1', 0.0)
        
        logger.info(f"   📊 Macro平均:")
        logger.info(f"      - 精确率 (Precision): {macro_precision:.4f}")
        logger.info(f"      - 召回率 (Recall): {macro_recall:.4f}")
        logger.info(f"      - F1值: {macro_f1:.4f}")
        
        # 加权平均指标
        weighted_precision = metrics.get(f'{task_name}_weighted_precision', 0.0)
        weighted_recall = metrics.get(f'{task_name}_weighted_recall', 0.0)
        weighted_f1 = metrics.get(f'{task_name}_weighted_f1', 0.0)
        
        logger.info(f"   ⚖️ Weighted平均:")
        logger.info(f"      - 精确率 (Precision): {weighted_precision:.4f}")
        logger.info(f"      - 召回率 (Recall): {weighted_recall:.4f}")
        logger.info(f"      - F1值: {weighted_f1:.4f}")
        
        # 每个类别的详细指标
        per_class_metrics = metrics.get(f'{task_name}_per_class', {})
        if per_class_metrics:
            logger.info(f"   🏷️ 各类别指标:")
            for label, class_metrics in per_class_metrics.items():
                precision = class_metrics['precision']
                recall = class_metrics['recall']
                f1 = class_metrics['f1']
                support = class_metrics['support']
                logger.info(f"      - {label}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, 样本数={support}")
        
        # 类别分布
        label_distribution = metrics.get(f'{task_name}_label_distribution', {})
        if label_distribution:
            logger.info(f"   📈 类别分布:")
            total_samples = sum(label_distribution.values())
            for label, count in sorted(label_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_samples * 100
                logger.info(f"      - {label}: {count} ({percentage:.1f}%)")
    
    def _print_confusion_matrices(self, metrics: Dict):
        """打印混淆矩阵"""
        confusion_matrices = metrics.get('confusion_matrices', {})
        
        for task_name, matrix in confusion_matrices.items():
            if not matrix:
                continue
                
            task_titles = {
                'transport': '🚛 运输方式混淆矩阵',
                'city': '🏙️ 城市预测混淆矩阵', 
                'province': '🗺️ 省份预测混淆矩阵'
            }
            
            logger.info(f"\n{task_titles.get(task_name, f'{task_name} 混淆矩阵')}:")
            
            # 获取所有标签
            all_labels = sorted(set(list(matrix.keys()) + [pred for pred_dict in matrix.values() for pred in pred_dict.keys()]))
            
            # 只显示前5个最常见的标签，避免输出过长
            if len(all_labels) > 5:
                # 按频率排序，选择前5个
                label_counts = {}
                for true_label, pred_dict in matrix.items():
                    label_counts[true_label] = label_counts.get(true_label, 0) + sum(pred_dict.values())
                top_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)[:5]
                all_labels = top_labels
                logger.info(f"   (显示top-{len(all_labels)}类别)")
            
            # 打印矩阵头部
            header = "真实\\预测".ljust(12)
            for label in all_labels:
                header += f"{label[:8]:>8}"
            logger.info(f"   {header}")
            
            # 打印矩阵内容
            for true_label in all_labels:
                row = f"{true_label[:10]:10}"
                for pred_label in all_labels:
                    count = matrix.get(true_label, {}).get(pred_label, 0)
                    row += f"{count:>8}"
                logger.info(f"   {row}")
    
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