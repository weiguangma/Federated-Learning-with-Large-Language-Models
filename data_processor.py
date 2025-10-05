#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextConverter:
    """文本转换器基类"""
    def __init__(self):
        self.missing_value_handlers = {
            'numeric': lambda x: f"约{np.random.randint(1, 100)}单位" if pd.isna(x) else str(x),
            'categorical': lambda x: "未知类型" if pd.isna(x) else str(x),
            'time': lambda x: "时间未知" if pd.isna(x) else str(x)
        }
    
    def safe_convert(self, value, value_type='categorical'):
        """安全转换值"""
        try:
            if pd.isna(value) or value is None or str(value).strip() == '':
                return self.missing_value_handlers[value_type](value)
            return str(value).strip()
        except:
            return "数据异常"

class PortTextConverter(TextConverter):
    """港口数据文本转换器"""
    def convert_to_text(self, row: Dict) -> str:
        try:
            # 提取关键信息
            cargo_type = self.safe_convert(row.get('货物类型', ''), 'categorical')
            weight = self.safe_convert(row.get('重量', ''), 'numeric')
            container_type = self.safe_convert(row.get('装箱方式', ''), 'categorical')
            trade_direction = self.safe_convert(row.get('贸易方向', ''), 'categorical')
            dest_port = self.safe_convert(row.get('目的港', ''), 'categorical')
            berth_time = self.safe_convert(row.get('等待时间', ''), 'numeric')
            yard_time = self.safe_convert(row.get('停留时间', ''), 'numeric')
            
            text = f"""<|港口专家|>
作为港口运营专家，我提供以下港口作业和货物信息：
货物类型：{cargo_type}。装箱方式：{container_type}。贸易方向：{trade_direction}。目的港：{dest_port}港。港口作业：靠泊正常，等待时间{berth_time}小时。堆场情况：堆场周转正常，停留{yard_time}小时。"""
            
            return text.strip()
        except Exception as e:
            logger.warning(f"港口数据转换失败: {e}")
            return "<|港口专家|>\n作为港口运营专家，当前港口数据暂时无法获取。"

class RailwayTextConverter(TextConverter):
    """铁路数据文本转换器"""
    def convert_to_text(self, row: Dict) -> str:
        try:
            # 提取关键信息
            scale = self.safe_convert(row.get('运输规模', ''), 'categorical')
            container_count = self.safe_convert(row.get('集装箱数量', ''), 'numeric')
            total_weight = self.safe_convert(row.get('总重量', ''), 'numeric')
            cost_level = self.safe_convert(row.get('费用水平', ''), 'categorical')
            total_cost = self.safe_convert(row.get('总费用', ''), 'numeric')
            rail_price = self.safe_convert(row.get('铁路报价', ''), 'numeric')
            distance = self.safe_convert(row.get('运输距离', ''), 'numeric')
            time_efficiency = self.safe_convert(row.get('运输时效', ''), 'categorical')
            estimated_time = self.safe_convert(row.get('预计时间', ''), 'numeric')
            route = self.safe_convert(row.get('运输路线', ''), 'categorical')
            
            text = f"""<|铁路专家|>  
作为铁路运输专家，我提供以下铁路运输和路线信息：
运输规模：{scale}，{container_count}个集装箱，总重{total_weight}吨。费用水平：{cost_level}，总费用{total_cost}元。铁路报价：{rail_price}元。运输距离：中距离运输，{distance}公里。运输时效：{time_efficiency}，预计{estimated_time}小时。运输路线：{route}。"""
            
            return text.strip()
        except Exception as e:
            logger.warning(f"铁路数据转换失败: {e}")
            return "<|铁路专家|>\n作为铁路运输专家，当前铁路数据暂时无法获取。"

class CustomsTextConverter(TextConverter):
    """海关数据文本转换器"""
    def convert_to_text(self, row: Dict) -> str:
        try:
            # 提取关键信息
            product_type = self.safe_convert(row.get('商品特征', ''), 'categorical')
            weight = self.safe_convert(row.get('重量', ''), 'numeric')
            
            text = f"""<|海关专家|>
作为海关业务专家，我提供以下贸易和清关信息：
商品特征：{product_type}，重量{weight}公斤。"""
            
            return text.strip()
        except Exception as e:
            logger.warning(f"海关数据转换失败: {e}")
            return "<|海关专家|>\n作为海关业务专家，当前海关数据暂时无法获取。"

class FederatedDataIntegrator:
    """联邦数据集成器 - 高性能版本"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.port_converter = PortTextConverter()
        self.railway_converter = RailwayTextConverter()
        self.customs_converter = CustomsTextConverter()
        
        # 数据文件路径
        self.data_files = {
            'port': self.data_dir / '港口数据_处理后.csv',
            'railway': self.data_dir / '铁路原始数据_补充与模拟_含公路特征.csv',
            'customs_railway': self.data_dir / '铁路海关模拟数据.csv',
            'customs_potential': self.data_dir / '潜在箱源_海关模拟_样例全.csv',
            'potential': self.data_dir / '潜在箱源模拟数据.csv'
        }
        
        logger.info(f"📁 数据目录: {self.data_dir}")
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据文件"""
        data = {}
        
        for name, file_path in self.data_files.items():
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    data[name] = df
                    logger.info(f"✅ 加载 {name}: {len(df)} 行")
                except Exception as e:
                    logger.warning(f"⚠️  加载 {name} 失败: {e}")
                    data[name] = pd.DataFrame()
            else:
                logger.warning(f"⚠️  文件不存在: {file_path}")
                data[name] = pd.DataFrame()
        
        return data
    
    def create_federated_samples(self, data: Dict[str, pd.DataFrame], sample_size: int = 2000) -> List[Dict]:
        """创建联邦样本 - 优化版本"""
        logger.info(f"🔧 开始创建 {sample_size} 个联邦样本...")
        
        # 使用港口数据作为主数据源
        port_data = data['port']
        railway_data = data['railway']
        customs_railway_data = data['customs_railway']
        customs_potential_data = data['customs_potential']
        
        if port_data.empty:
            logger.error("❌ 港口数据为空，无法创建样本")
            return []
        
        # 创建高效索引 - O(1)查找
        logger.info("🔍 创建高效索引...")
        
        # 确保CNTR列为字符串类型
        if 'CNTR' in railway_data.columns:
            railway_data['CNTR'] = railway_data['CNTR'].astype(str)
        if 'XH' in railway_data.columns:
            railway_data['XH'] = railway_data['XH'].astype(str)
        if 'CNTR' in customs_railway_data.columns:
            customs_railway_data['CNTR'] = customs_railway_data['CNTR'].astype(str)
        if 'CNTR' in port_data.columns:
            port_data['CNTR'] = port_data['CNTR'].astype(str)
            
        # 创建哈希索引 - 处理重复索引
        railway_cntr_index = {}
        if 'CNTR' in railway_data.columns:
            # 去重后创建索引
            railway_data_dedup = railway_data.drop_duplicates(subset=['CNTR'], keep='first')
            railway_cntr_index = railway_data_dedup.set_index('CNTR').to_dict('index')
        
        railway_xh_index = {}
        if 'XH' in railway_data.columns:
            # 去重后创建索引
            railway_data_dedup_xh = railway_data.drop_duplicates(subset=['XH'], keep='first')
            railway_xh_index = railway_data_dedup_xh.set_index('XH').to_dict('index')
        
        customs_cntr_index = {}
        if 'CNTR' in customs_railway_data.columns:
            # 去重后创建索引
            customs_data_dedup = customs_railway_data.drop_duplicates(subset=['CNTR'], keep='first')
            customs_cntr_index = customs_data_dedup.set_index('CNTR').to_dict('index')
        
        logger.info(f"📊 索引统计: 铁路CNTR={len(railway_cntr_index)}, 铁路XH={len(railway_xh_index)}, 海关CNTR={len(customs_cntr_index)}")
        
        federated_samples = []
        start_time = time.time()
        
        # 限制样本数量
        total_samples = min(sample_size, len(port_data))
        
        for idx in tqdm(range(total_samples), desc="创建联邦样本"):
            try:
                port_row = port_data.iloc[idx]
                
                # 安全提取container_id
                try:
                    cntr_value = port_row.get('CNTR', f'sample_{idx}')
                    if hasattr(cntr_value, 'iloc'):
                        cntr_value = cntr_value.iloc[0] if len(cntr_value) > 0 else f'sample_{idx}'
                    if cntr_value is None or pd.isna(cntr_value) or str(cntr_value).strip() == '':
                        container_id = f'sample_{idx}'
                    else:
                        container_id = str(cntr_value).strip()
                except Exception:
                    container_id = f'sample_{idx}'
                
                # O(1)查找匹配的铁路数据
                railway_row = railway_cntr_index.get(container_id)
                if railway_row is None:
                    railway_row = railway_xh_index.get(container_id)
                
                # O(1)查找匹配的海关数据
                customs_row = customs_cntr_index.get(container_id)
                
                # 转换为字典格式
                try:
                    if port_row is not None and hasattr(port_row, 'to_dict'):
                        port_dict = port_row.to_dict()
                    elif port_row is not None and hasattr(port_row, 'iloc'):
                        port_dict = dict(port_row)
                    else:
                        port_dict = None
                except Exception:
                    port_dict = None
                
                # 生成文本描述
                port_text = self.port_converter.convert_to_text(port_dict) if port_dict else "<|港口专家|>\n港口数据暂时无法获取。"
                railway_text = self.railway_converter.convert_to_text(railway_row) if railway_row else "<|铁路专家|>\n铁路数据暂时无法获取。"
                customs_text = self.customs_converter.convert_to_text(customs_row) if customs_row else "<|海关专家|>\n海关数据暂时无法获取。"
                
                # 生成决策输出
                output_text = self._generate_output(port_dict, railway_row, customs_row)
                
                # 创建联邦样本
                sample = {
                    'container_id': container_id,
                    'port_text': port_text,
                    'railway_text': railway_text,
                    'customs_text': customs_text,
                    'output': output_text,
                    'sample_id': f'federated_{idx:08d}'
                }
                
                federated_samples.append(sample)
                
            except Exception as e:
                logger.warning(f"创建联邦样本失败 [{idx}]: {e}")
                continue
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if len(federated_samples) > 0:
            avg_time = processing_time / len(federated_samples)
            logger.info(f"✅ 联邦样本创建完成: {len(federated_samples)} 个")
            logger.info(f"⏱️  处理时间: {processing_time:.2f}秒, 平均每样本: {avg_time:.4f}秒")
        else:
            logger.error("❌ 没有成功创建任何联邦样本")
        
        return federated_samples
    
    def _generate_output(self, port_data: Dict, railway_data: Dict, customs_data: Dict) -> str:
        """生成决策输出"""
        try:
            # 简单的决策逻辑
            transport_mode = "铁路运输"
            destination = "未知"
            province = "未知"
            
            # 从铁路数据提取目的地
            if railway_data and '运输路线' in railway_data:
                route = str(railway_data['运输路线'])
                if '到' in route:
                    destination = route.split('到')[-1].replace('站', '').strip()
            
            output = f"""基于多源信息综合分析：

**运输方式选择**: {transport_mode}
**目的地城市**: {destination}  
**目的地省份**: {province}

**决策理由**:
1. **多源数据融合**: 综合港口作业效率、铁路运输成本、海关清关时效等关键因素
2. **运输方式优选**: 选择{transport_mode}基于成本效益和时效性平衡
3. **目的地预测**: 根据货物特征和贸易流向，确定目的地为{province}{destination}
4. **风险评估**: 考虑运输路径、天气条件、政策影响等风险因素
5. **优化建议**: 建议采用多式联运以提高整体效率"""
            
            return output
        except Exception as e:
            logger.warning(f"生成输出失败: {e}")
            return "基于当前可用信息，建议采用综合运输方案。"

class QwenFederatedFormatter:
    """Qwen联邦格式化器"""
    
    def __init__(self):
        self.separator = " <|object_ref_start|> "
        self.end_separator = " <|object_ref_end|> "
    
    def is_valid_sample(self, sample: Dict) -> bool:
        """检查样本是否有效"""
        try:
            # 检查必要字段是否存在
            required_fields = ['port_text', 'railway_text', 'customs_text', 'output']
            for field in required_fields:
                if field not in sample or not sample[field]:
                    return False
            
            # 检查客户端信息是否有效（不能是"数据暂时无法获取"）
            invalid_phrases = [
                "数据暂时无法获取",
                "暂时无法获取",
                "无法获取",
                "数据缺失",
                "信息不完整"
            ]
            
            for field in ['port_text', 'railway_text', 'customs_text']:
                text = sample[field].lower()
                if any(phrase in text for phrase in invalid_phrases):
                    return False
            
            # 检查港口信息是否完整
            port_text = sample['port_text']
            required_port_info = ['货物类型：', '装箱方式：', '贸易方向：', '目的港：']
            for info in required_port_info:
                if info in port_text:
                    # 检查冒号后是否有实际内容（不能只是空格或"。"）
                    after_colon = port_text.split(info)[1].split('。')[0].strip()
                    if not after_colon or after_colon in ['', '无', '未知', '-']:
                        return False
            
            # 检查输出中的目的地信息
            output_text = sample['output']
            if '目的地城市**: 未知' in output_text or '目的地省份**: 未知' in output_text:
                return False
            
            # 检查是否包含有效的运输方式
            if '运输方式选择**:' not in output_text:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"样本验证失败: {e}")
            return False
    
    def format_to_qwen(self, sample: Dict) -> Dict[str, str]:
        """转换为Qwen训练格式"""
        try:
            # 首先检查样本是否有效
            if not self.is_valid_sample(sample):
                return None
            
            # 构建instruction
            server_instruction = "你是一个智能物流决策系统。基于多个客户端提供的专业信息，请综合分析并做出最优的运输决策。\n\n请根据以下多源数据信息，选择最适合的运输方式，预测目的地，并提供详细的决策理由："
            
            instruction = (
                server_instruction +
                self.separator + sample['port_text'] + self.end_separator +
                self.separator + sample['railway_text'] + self.end_separator +
                self.separator + sample['customs_text'] + self.end_separator
            )
            
            return {
                "instruction": instruction,
                "input": "",
                "output": sample['output'],
                "sample_id": sample['sample_id']
            }
        except Exception as e:
            logger.error(f"格式化样本失败: {e}")
            return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="联邦学习数据处理")
    parser.add_argument("--data_dir", type=str, 
                       default="/root/autodl-tmp/Federated_learning/code_v01/verify_data",
                       help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="./data/qwen_processed",
                       help="输出目录")
    parser.add_argument("--sample_size", type=int, default=2000,
                       help="生成样本数量")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="训练集比例 (默认: 0.8)")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 数据集成
        logger.info("🚀 开始联邦数据处理...")
        integrator = FederatedDataIntegrator(args.data_dir)
        data = integrator.load_data()
        
        # 2. 创建联邦样本（生成更多样本以确保过滤后有足够的有效样本）
        # 预估过滤率，生成更多样本
        estimated_filter_rate = 0.3  # 预估30%的样本会被过滤
        target_samples = int(args.sample_size / (1 - estimated_filter_rate))
        
        logger.info(f"🎯 目标有效样本数: {args.sample_size}")
        logger.info(f"📊 考虑过滤率，生成样本数: {target_samples}")
        
        federated_samples = integrator.create_federated_samples(data, target_samples)
        
        if not federated_samples:
            logger.error("❌ 没有生成任何样本")
            return False
        
        # 3. 格式化为Qwen格式并过滤无效样本
        logger.info("📝 转换为Qwen训练格式并过滤无效样本...")
        formatter = QwenFederatedFormatter()
        qwen_samples = []
        filtered_count = 0
        
        for sample in tqdm(federated_samples, desc="格式化和过滤样本"):
            qwen_sample = formatter.format_to_qwen(sample)
            if qwen_sample:
                qwen_samples.append(qwen_sample)
            else:
                filtered_count += 1
        
        logger.info(f"📊 样本过滤结果:")
        logger.info(f"   - 原始样本数: {len(federated_samples)}")
        logger.info(f"   - 过滤掉的样本: {filtered_count}")
        logger.info(f"   - 有效样本数: {len(qwen_samples)}")
        logger.info(f"   - 过滤率: {filtered_count/len(federated_samples):.1%}")
        
        # 如果有效样本数量不足，生成更多
        if len(qwen_samples) < args.sample_size:
            logger.warning(f"⚠️ 有效样本数量不足 ({len(qwen_samples)} < {args.sample_size})")
            logger.info("🔄 生成更多样本以达到目标数量...")
            
            additional_needed = args.sample_size - len(qwen_samples)
            additional_raw = int(additional_needed / (1 - filtered_count/len(federated_samples))) + 100
            
            additional_samples = integrator.create_federated_samples(data, additional_raw)
            for sample in tqdm(additional_samples, desc="生成额外样本"):
                if len(qwen_samples) >= args.sample_size:
                    break
                qwen_sample = formatter.format_to_qwen(sample)
                if qwen_sample:
                    qwen_samples.append(qwen_sample)
        
        # 如果样本过多，随机选择目标数量
        if len(qwen_samples) > args.sample_size:
            import random
            random.seed(args.random_seed)
            qwen_samples = random.sample(qwen_samples, args.sample_size)
            
        logger.info(f"✅ 最终有效样本数: {len(qwen_samples)}")
        
        # 4. 分割训练集和测试集
        logger.info(f"🔀 分割训练集和测试集...")
        
        import random
        random.seed(args.random_seed)
        
        # 随机打乱样本
        shuffled_samples = qwen_samples.copy()
        random.shuffle(shuffled_samples)
        
        # 计算分割点
        total_samples = len(shuffled_samples)
        train_size = int(total_samples * args.train_ratio)
        test_size = total_samples - train_size
        
        train_samples = shuffled_samples[:train_size]
        test_samples = shuffled_samples[train_size:]
        
        logger.info(f"📊 数据分割结果:")
        logger.info(f"   - 总样本数: {total_samples}")
        logger.info(f"   - 训练集: {len(train_samples)} ({args.train_ratio:.1%})")
        logger.info(f"   - 测试集: {len(test_samples)} ({1-args.train_ratio:.1%})")
        
        # 5. 保存训练集
        train_file = output_dir / "qwen_federated_train.jsonl"
        logger.info(f"💾 保存训练集到: {train_file}")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 6. 保存测试集
        test_file = output_dir / "qwen_federated_test.jsonl"
        logger.info(f"💾 保存测试集到: {test_file}")
        
        with open(test_file, 'w', encoding='utf-8') as f:
            for sample in test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 7. 保存数据集统计信息
        dataset_stats = {
            "total_samples": total_samples,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "train_ratio": args.train_ratio,
            "random_seed": args.random_seed,
            "train_file": str(train_file),
            "test_file": str(test_file),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        stats_file = output_dir / "dataset_split_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"🎉 数据处理完成！")
        logger.info(f"   - 原始样本: {len(federated_samples)}")
        logger.info(f"   - Qwen格式样本: {len(qwen_samples)}")
        logger.info(f"   - 训练集文件: {train_file} ({train_file.stat().st_size / 1024 / 1024:.2f} MB)")
        logger.info(f"   - 测试集文件: {test_file} ({test_file.stat().st_size / 1024 / 1024:.2f} MB)")
        logger.info(f"   - 统计信息: {stats_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
