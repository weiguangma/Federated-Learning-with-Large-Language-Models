# /********************************************************************************
#  * @Author: zhangqiuhong
#  * @Date: 2025-09-20
#  * @Description: 增强版联邦学习数据处理器 - 使用高质量完整数据源
#  ********************************************************************************/

import pandas as pd
import numpy as np
import json
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedFederatedDataProcessor:
    """增强版联邦学习数据处理器 - 使用高质量完整数据源"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.main_data_file = self.data_dir / '潜在箱源_海关模拟_样例全.csv'
        logger.info(f"📁 数据目录: {self.data_dir}")
        logger.info(f"🎯 使用高质量数据源: {self.main_data_file.name}")

    def load_enhanced_data(self) -> pd.DataFrame:
        """加载增强版完整数据"""
        try:
            if not self.main_data_file.exists():
                raise FileNotFoundError(f"数据文件不存在: {self.main_data_file}")

            logger.info("📊 加载高质量完整数据...")
            df = pd.read_csv(self.main_data_file, encoding='utf-8')
            logger.info(f"✅ 成功加载数据: {len(df)} 行, {len(df.columns)} 列")

            # 数据质量检查
            self._check_data_quality(df)

            return df

        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise

    def _check_data_quality(self, df: pd.DataFrame):
        """检查数据质量"""
        logger.info("🔍 检查数据质量...")

        key_columns = ['CNTR', 'PM', 'FZHZM',
                       'DZHZM', '目的地省份', '目的地城市', 'CARGO_WGT']

        quality_report = {}
        for col in key_columns:
            if col in df.columns:
                non_null_ratio = df[col].notna().mean()
                unique_count = df[col].nunique()
                quality_report[col] = {
                    'non_null_ratio': non_null_ratio,
                    'unique_count': unique_count
                }
                logger.info(
                    f"  {col}: 非空率={non_null_ratio:.2%}, 唯一值={unique_count}")
            else:
                logger.warning(f"  ⚠️  关键列缺失: {col}")

        return quality_report

    def create_enhanced_samples(self, df: pd.DataFrame, sample_size: int = 2000,
                                railway_ratio: float = 0.6) -> List[Dict]:
        """创建增强版联邦样本，控制运输方式比例"""
        logger.info(f"🚀 开始创建 {sample_size} 个增强版联邦样本...")
        logger.info(
            f"📊 运输方式比例设置: 铁路运输 {railway_ratio:.1%}, 非铁路运输 {1-railway_ratio:.1%}")

        # 过滤有效数据
        valid_df = self._filter_valid_data(df)
        logger.info(f"📊 过滤后有效数据: {len(valid_df)} 行")

        if len(valid_df) < sample_size:
            logger.warning(f"⚠️  有效数据不足，调整样本数量: {len(valid_df)}")
            sample_size = len(valid_df)

        # 计算各运输方式的样本数量
        railway_samples = int(sample_size * railway_ratio)
        non_railway_samples = sample_size - railway_samples

        logger.info(
            f"📊 样本分配: 铁路运输 {railway_samples} 个, 非铁路运输 {non_railway_samples} 个")

        # 随机采样
        sampled_df = valid_df.sample(
            n=sample_size, random_state=42).reset_index(drop=True)

        # 创建联邦样本
        samples = []
        railway_count = 0
        non_railway_count = 0

        for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="创建增强样本"):
            try:
                # 根据比例控制运输方式
                if railway_count < railway_samples:
                    transport_mode = 'railway'
                    railway_count += 1
                elif non_railway_count < non_railway_samples:
                    transport_mode = 'non_railway'
                    non_railway_count += 1
                else:
                    # 如果两种方式都已满额，随机选择
                    transport_mode = 'railway' if random.random() < 0.5 else 'non_railway'

                sample = self._create_single_sample(row, idx, transport_mode)
                if sample:
                    samples.append(sample)
            except Exception as e:
                logger.warning(f"样本 {idx} 创建失败: {e}")
                continue

        # 统计最终的运输方式分布
        railway_final = sum(
            1 for s in samples if '**运输方式选择**: 铁路运输' in s.get('output', ''))
        non_railway_final = sum(
            1 for s in samples if '**运输方式选择**: 公路运输' in s.get('output', ''))
        other_final = len(samples) - railway_final - non_railway_final

        logger.info(f"✅ 成功创建 {len(samples)} 个增强版联邦样本")
        logger.info(f"📊 最终运输方式分布: 铁路运输 {railway_final} ({railway_final/len(samples):.1%}), "
                    f"非铁路运输 {non_railway_final} ({non_railway_final/len(samples):.1%})")

        return samples

    def _filter_valid_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """过滤有效数据"""
        logger.info("🔍 过滤有效数据...")

        # 基本过滤条件
        conditions = [
            df['CNTR'].notna(),
            df['PM'].notna(),
            df['FZHZM'].notna(),
            df['DZHZM'].notna(),
            df['目的地省份'].notna(),
            df['目的地城市'].notna(),
            df['CARGO_WGT'].notna(),
            df['CARGO_WGT'] > 0
        ]

        # 组合所有条件
        valid_mask = pd.concat(conditions, axis=1).all(axis=1)
        valid_df = df[valid_mask].copy()

        # 额外的质量过滤
        valid_df = valid_df[
            (valid_df['目的地省份'] != '未知') &
            (valid_df['目的地城市'] != '未知') &
            (valid_df['PM'] != '') &
            (valid_df['FZHZM'] != '') &
            (valid_df['DZHZM'] != '')
        ]

        logger.info(
            f"📊 过滤结果: {len(df)} → {len(valid_df)} 行 (保留率: {len(valid_df)/len(df):.2%})")
        return valid_df

    def _create_single_sample(self, row: pd.Series, sample_idx: int, transport_mode: str = 'railway') -> Optional[Dict]:
        """创建单个增强样本"""
        try:
            # 港口专家信息
            port_text = self._create_port_expert_text(row)

            # 铁路专家信息
            railway_text = self._create_railway_expert_text(
                row, transport_mode)

            # 海关专家信息
            customs_text = self._create_customs_expert_text(row)

            # 决策输出
            output_text = self._create_decision_output(row, transport_mode)

            return {
                'sample_id': f"enhanced_{sample_idx:06d}",
                'port_text': port_text,
                'railway_text': railway_text,
                'customs_text': customs_text,
                'output': output_text,
                'cntr': row.get('CNTR', ''),
                'cargo_type': row.get('PM', ''),
                'origin': row.get('FZHZM', ''),
                'destination': row.get('DZHZM', ''),
                'dest_province': row.get('目的地省份', ''),
                'dest_city': row.get('目的地城市', '')
            }

        except Exception as e:
            logger.warning(f"创建样本失败: {e}")
            return None

    def _create_port_expert_text(self, row: pd.Series) -> str:
        """创建港口专家文本"""
        cargo_type = row.get('PM', '未知货物')
        cargo_weight = row.get('CARGO_WGT', 0)
        container_size = row.get('CNTR_SIZ_COD', '20')
        container_type = row.get('CNTR_TYP_COD', 'GP')
        load_port = row.get('LOAD_PORT_COD', 'CNNSA')
        disc_port = row.get('DISC_PORT_COD', 'CNNSA')
        dest_port = row.get('DEST_PORT_COD', 'CNXIN')

        # 港口作业时间信息（智能选择数据源）
        berth_time = None

        # 优先使用WAIT_HOURS字段（潜在箱源数据）
        if 'WAIT_HOURS' in row and pd.notna(row.get('WAIT_HOURS')):
            berth_time = float(row['WAIT_HOURS']) * \
                random.uniform(0.8, 1.2)  # 添加±20%随机性
        # 备选使用ETA_to_BerthTime字段（港口数据）
        elif 'ETA_to_BerthTime' in row and pd.notna(row.get('ETA_to_BerthTime')):
            eta_berth = float(row['ETA_to_BerthTime'])
            # 处理负值和异常值
            if eta_berth > 0 and eta_berth < 100:
                berth_time = eta_berth * random.uniform(0.8, 1.2)

        # 如果都没有有效数据，使用合理的随机值
        if berth_time is None or berth_time <= 0 or berth_time > 100:
            berth_time = random.uniform(8, 48)

        # 堆场停留时间：根据货物类型和港口作业情况随机生成
        yard_stay = random.uniform(24, 72)

        port_text = f"""<|港口专家|>
作为港口运营专家，我提供以下港口作业和货物信息：
货物类型：{cargo_type}。装箱方式：{container_size}尺寸{container_type}型集装箱。贸易方向：进口。目的港：{dest_port}港。港口作业：靠泊正常，等待时间{berth_time:.1f}小时。堆场情况：堆场周转正常，停留{yard_stay:.1f}小时。货物重量：{cargo_weight}公斤。装货港：{load_port}，卸货港：{disc_port}。"""

        return port_text

    def _create_railway_expert_text(self, row: pd.Series, transport_mode: str = 'railway') -> str:
        """创建铁路专家文本"""
        origin_station = row.get('FZHZM', '南沙港')
        dest_station = row.get('DZHZM', '三原')
        cargo_type = row.get('PM', '面粉')

        if transport_mode == 'railway':
            # 铁路运输模式
            train_no = row.get('XH', f"G{random.randint(1000, 9999)}")
            transit_hours = row.get('TRANSIT_HOURS', 24)
            rail_price = row.get('95306_PRICE', 3500)

            railway_text = f"""<|铁路专家|>
作为铁路运输专家，我提供以下铁路运输信息：
列车车次：{train_no}，发站：{origin_station}，到站：{dest_station}。货物品名：{cargo_type}。预计运输时间：{transit_hours:.1f}小时。铁路运价：{rail_price}元。运输状态：正常运行，无延误。线路条件：干线直达，运能充足。"""
        else:
            # 非铁路运输模式（公路运输）
            distance = random.randint(300, 1200)  # 公路距离
            road_hours = distance / random.randint(60, 80)  # 按速度计算时间
            road_price = distance * random.uniform(2.5, 4.0)  # 公路运价
            transit_hours = row.get('TRANSIT_HOURS', 24)  # 获取铁路运输时间作为对比

            railway_text = f"""<|铁路专家|>
作为铁路运输专家，我提供以下铁路运输分析：
铁路线路：{origin_station}至{dest_station}线路当前运能紧张。货物品名：{cargo_type}。铁路运输时效：约{transit_hours:.1f}小时，但发车班次有限。建议考虑公路运输替代方案：预计公路距离{distance}公里，运输时间{road_hours:.1f}小时，运价约{road_price:.0f}元。"""

        return railway_text

    def _create_customs_expert_text(self, row: pd.Series) -> str:
        """创建海关专家文本"""
        clearance_time = row.get('customs_clearance_time_days', 2)
        risk_level = row.get('customs_risk_level', 'LOW')
        inspection_prob = row.get('inspection_probability', 0.1)
        dest_province = row.get('目的地省份', '陕西省')
        dest_city = row.get('目的地城市', '咸阳市')

        risk_level_cn = {'LOW': '低风险', 'MEDIUM': '中风险',
                         'HIGH': '高风险'}.get(risk_level, '低风险')

        customs_text = f"""<|海关专家|>
作为海关业务专家，我提供以下海关清关信息：
清关状态：正常通关。预计清关时间：{clearance_time}天。风险等级：{risk_level_cn}。查验概率：{inspection_prob:.1%}。目的地：{dest_province}{dest_city}。贸易合规：符合相关法规要求。单证齐全：进口许可、原产地证明等单证完备。"""

        return customs_text

    def _create_decision_output(self, row: pd.Series, transport_mode: str = 'railway') -> str:
        """创建决策输出"""
        dest_province = row.get('目的地省份', '陕西省')
        dest_city = row.get('目的地城市', '咸阳市')
        cargo_type = row.get('PM', '面粉')
        origin_station = row.get('FZHZM', '南沙港')
        dest_station = row.get('DZHZM', '三原')

        # 根据传入的运输方式参数确定运输方式
        if transport_mode == 'railway':
            transport_mode_cn = "铁路运输"
            mode_reason = "基于铁路直达线路和成本优势"
            optimization_detail = f"{origin_station}至{dest_station}铁路直达，运输效率高"
        else:
            transport_mode_cn = "公路运输"
            mode_reason = "基于灵活性和时效性考虑"
            optimization_detail = f"{origin_station}至{dest_province}{dest_city}公路运输，门到门服务"

        output = f"""基于多源信息综合分析：

**运输方式选择**: {transport_mode_cn}
**目的地城市**: {dest_city}
**目的地省份**: {dest_province}

**决策理由**:
1. **多源数据融合**: 综合港口作业效率、铁路运输成本、海关清关时效等关键因素
2. **运输方式优选**: 选择{transport_mode_cn}{mode_reason}
3. **目的地预测**: 根据货物特征和贸易流向，确定目的地为{dest_province}{dest_city}
4. **货物适配**: {cargo_type}类货物适合当前运输方案
5. **路径优化**: {optimization_detail}"""

        return output


class EnhancedQwenFormatter:
    """增强版Qwen格式化器"""

    def __init__(self):
        self.separator = " <|object_ref_start|> "
        self.end_separator = " <|object_ref_end|> "

    def format_to_qwen(self, sample: Dict) -> Dict[str, str]:
        """转换为Qwen训练格式"""
        try:
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
    parser = argparse.ArgumentParser(description="增强版联邦学习数据处理")
    parser.add_argument("--data_dir", type=str,
                        default="/root/autodl-tmp/Federated_learning/code_v01/verify_data",
                        help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="./data/enhanced_processed",
                        help="输出目录")
    parser.add_argument("--sample_size", type=int, default=2000,
                        help="生成样本数量")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例 (默认: 0.8)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--railway_ratio", type=float, default=0.6,
                        help="铁路运输比例 (默认: 0.6，即60%铁路运输，40%公路运输)")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("🚀 开始增强版联邦数据处理...")

        # 1. 初始化处理器
        processor = EnhancedFederatedDataProcessor(args.data_dir)

        # 2. 加载高质量数据
        df = processor.load_enhanced_data()

        # 3. 创建增强样本（控制运输方式比例）
        samples = processor.create_enhanced_samples(
            df, args.sample_size, args.railway_ratio)

        if not samples:
            logger.error("❌ 没有生成任何样本")
            return False

        # 4. 格式化为Qwen格式
        logger.info("📝 转换为Qwen训练格式...")
        formatter = EnhancedQwenFormatter()
        qwen_samples = []

        for sample in tqdm(samples, desc="格式化样本"):
            qwen_sample = formatter.format_to_qwen(sample)
            if qwen_sample:
                qwen_samples.append(qwen_sample)

        logger.info(f"✅ 成功格式化 {len(qwen_samples)} 个样本")

        # 5. 分割训练集和测试集
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
        logger.info(
            f"   - 测试集: {len(test_samples)} ({1-args.train_ratio:.1%})")

        # 6. 保存训练集
        train_file = output_dir / "enhanced_qwen_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 7. 保存测试集
        test_file = output_dir / "enhanced_qwen_test.jsonl"
        with open(test_file, 'w', encoding='utf-8') as f:
            for sample in test_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 8. 保存统计信息
        stats = {
            'total_samples': total_samples,
            'train_samples': len(train_samples),
            'test_samples': len(test_samples),
            'train_ratio': args.train_ratio,
            'random_seed': args.random_seed,
            'data_source': processor.main_data_file.name,
            'processing_time': pd.Timestamp.now().isoformat()
        }

        stats_file = output_dir / "enhanced_processing_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 数据保存完成:")
        logger.info(f"   - 训练集: {train_file}")
        logger.info(f"   - 测试集: {test_file}")
        logger.info(f"   - 统计信息: {stats_file}")
        logger.info("🎉 增强版数据处理完成!")

        return True

    except Exception as e:
        logger.error(f"❌ 数据处理失败: {e}")
        return False


if __name__ == "__main__":
    main()
