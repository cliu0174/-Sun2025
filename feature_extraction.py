'''
Author: Fujin Wang
Date: 2024/5/28
Github: https://github.com/wang-fujin

Description:
该代码用于从HUST电池数据集中提取特征，生成CSV文件供机器学习建模使用。
特征包括CC和CV阶段的电压、电流统计特征，以及容量、充电时间等信息。

Version:
- FeatureExtractor: 基础版本
- RobustFeatureExtractor: 改进版本(使用稳健统计方法和数据清洗策略)
'''
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import medfilt
from HUST_dataloader import Battery


class FeatureExtractor:
    """电池充电数据特征提取器"""

    def __init__(self, battery):
        """
        初始化特征提取器
        :param battery: Battery对象
        """
        self.battery = battery

    def calculate_entropy(self, data, bins=50):
        """
        计算数据的信息熵
        :param data: array-like，输入数据
        :param bins: int，直方图分箱数
        :return: float，熵值
        """
        data = np.array(data)
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        # 计算概率密度
        prob = hist * bin_width
        # 过滤掉零概率，避免log(0)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob))
        return entropy

    def calculate_slope(self, time, value):
        """
        计算变化斜率（线性拟合）
        :param time: array-like，时间序列
        :param value: array-like，数值序列
        :return: float，斜率
        """
        time = np.array(time)
        value = np.array(value)
        if len(time) < 2:
            return 0.0
        # 线性拟合
        slope, intercept = np.polyfit(time, value, 1)
        return slope

    def extract_voltage_features(self, voltage_series):
        """
        提取电压统计特征
        :param voltage_series: Series，电压数据
        :return: dict，特征字典
        """
        voltage = voltage_series.values
        features = {
            'voltage mean': np.mean(voltage),
            'voltage std': np.std(voltage),
            'voltage kurtosis': stats.kurtosis(voltage),
            'voltage skewness': stats.skew(voltage)
        }
        return features

    def extract_current_features(self, current_series):
        """
        提取电流统计特征
        :param current_series: Series，电流数据
        :return: dict，特征字典
        """
        current = current_series.values / 1000  # 转换为A
        features = {
            'current mean': np.mean(current),
            'current std': np.std(current),
            'current kurtosis': stats.kurtosis(current),
            'current skewness': stats.skew(current)
        }
        return features

    def extract_CC_features(self, cycle):
        """
        提取CC阶段特征
        :param cycle: int，循环次数
        :return: dict，CC阶段特征
        """
        CC_df = self.battery.get_CC_stage(cycle)

        if len(CC_df) == 0:
            return None

        features = {}

        # 电压特征
        voltage_features = self.extract_voltage_features(CC_df['Voltage (V)'])
        features.update(voltage_features)

        # CC阶段容量和时间
        features['CC Q'] = (CC_df['Capacity (mAh)'].max() - CC_df['Capacity (mAh)'].min()) / 1000  # 转换为Ah - 使用增量容量
        features['CC charge time'] = int(CC_df['Time (s)'].max() - CC_df['Time (s)'].min())

        # 电压变化斜率
        time = CC_df['Time (s)'].values
        voltage = CC_df['Voltage (V)'].values
        features['voltage slope'] = self.calculate_slope(time, voltage)

        # 电压熵
        features['voltage entropy'] = self.calculate_entropy(voltage)

        return features

    def extract_CV_features(self, cycle):
        """
        提取CV阶段特征
        :param cycle: int，循环次数
        :return: dict，CV阶段特征
        """
        CV_df = self.battery.get_CV_stage(cycle)

        if len(CV_df) == 0:
            return None

        features = {}

        # 电流特征
        current_features = self.extract_current_features(CV_df['Current (mA)'])
        features.update(current_features)

        # CV阶段容量和时间
        features['CV Q'] = (CV_df['Capacity (mAh)'].max() - CV_df['Capacity (mAh)'].min()) / 1000  # 转换为Ah - 使用增量容量
        features['CV charge time'] = int(CV_df['Time (s)'].max() - CV_df['Time (s)'].min())

        # 电流变化斜率（CV阶段电流逐渐衰减）
        time = CV_df['Time (s)'].values
        current = CV_df['Current (mA)'].values / 1000
        features['current slope'] = self.calculate_slope(time, current)

        # 电流熵
        features['current entropy'] = self.calculate_entropy(current)

        return features

    def extract_one_cycle_features(self, cycle):
        """
        提取单个循环的完整特征
        :param cycle: int，循环次数
        :return: dict，特征字典
        """
        # CC阶段特征
        CC_features = self.extract_CC_features(cycle)
        if CC_features is None:
            return None

        # CV阶段特征
        CV_features = self.extract_CV_features(cycle)
        if CV_features is None:
            return None

        # 合并特征
        features = {**CC_features, **CV_features}

        # 添加总容量（目标变量）
        cycle_df = self.battery.get_cycle(cycle)
        total_capacity = cycle_df['Capacity (mAh)'].max() / 1000  # 转换为Ah
        features['capacity'] = total_capacity

        return features

    def extract_all_cycles(self):
        """
        提取所有循环的特征
        :return: DataFrame，特征数据框
        """
        all_features = []

        print(f"开始提取电池 {self.battery.battery_id} 的特征...")

        for cycle in range(1, self.battery.cycle_life + 1):
            try:
                features = self.extract_one_cycle_features(cycle)
                if features is not None:
                    all_features.append(features)

                # 显示进度
                if cycle % 100 == 0:
                    print(f"已处理 {cycle}/{self.battery.cycle_life} 个循环")

            except Exception as e:
                print(f"循环 {cycle} 提取特征失败: {str(e)}")
                continue

        print(f"特征提取完成！共提取 {len(all_features)} 个循环的特征")

        # 转换为DataFrame
        df = pd.DataFrame(all_features)

        # 确保列的顺序
        column_order = [
            'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
            'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
            'current mean', 'current std', 'current kurtosis', 'current skewness',
            'CV Q', 'CV charge time', 'current slope', 'current entropy',
            'capacity'
        ]
        df = df[column_order]

        return df


class RobustFeatureExtractor(FeatureExtractor):
    """
    改进的鲁棒特征提取器

    主要改进:
    1. 使用3-sigma过滤统计特征中的异常值
    2. 自适应熵值计算(Freedman-Diaconis规则)
    3. 添加数据质量评估
    """

    def __init__(self, battery, config=None):
        """
        初始化鲁棒特征提取器
        :param battery: Battery对象
        :param config: 配置字典
        """
        super().__init__(battery)
        self.config = config or {
            'use_robust_statistics': True,   # 使用稳健统计(3-sigma过滤)
            'auto_entropy_bins': True,       # 自动选择熵bins
            'outlier_threshold': 3.0,        # 异常值阈值(几倍std)
            'min_valid_ratio': 0.8          # 过滤后最小保留比例
        }

    def extract_voltage_features(self, voltage_series):
        """
        使用稳健统计方法提取电压特征
        :param voltage_series: Series，电压数据
        :return: dict，特征字典
        """
        voltage = voltage_series.values

        if self.config['use_robust_statistics']:
            # 3-sigma过滤异常值
            mean, std = voltage.mean(), voltage.std()
            threshold = self.config['outlier_threshold']
            mask = np.abs(voltage - mean) < threshold * std
            voltage_clean = voltage[mask]

            # 如果过滤掉太多数据,说明数据异常,使用原始数据
            if len(voltage_clean) < len(voltage) * self.config['min_valid_ratio']:
                voltage_clean = voltage
                outlier_ratio = 0.0
            else:
                outlier_ratio = 1 - len(voltage_clean) / len(voltage)
        else:
            voltage_clean = voltage
            outlier_ratio = 0.0

        features = {
            'voltage mean': np.mean(voltage_clean),
            'voltage std': np.std(voltage_clean),
            'voltage kurtosis': stats.kurtosis(voltage_clean),
            'voltage skewness': stats.skew(voltage_clean)
        }

        return features

    def extract_current_features(self, current_series):
        """
        使用稳健统计方法提取电流特征
        :param current_series: Series，电流数据
        :return: dict，特征字典
        """
        current = current_series.values / 1000  # 转换为A

        if self.config['use_robust_statistics']:
            # 3-sigma过滤异常值
            mean, std = current.mean(), current.std()
            threshold = self.config['outlier_threshold']
            mask = np.abs(current - mean) < threshold * std
            current_clean = current[mask]

            # 如果过滤掉太多数据,说明数据异常,使用原始数据
            if len(current_clean) < len(current) * self.config['min_valid_ratio']:
                current_clean = current
        else:
            current_clean = current

        features = {
            'current mean': np.mean(current_clean),
            'current std': np.std(current_clean),
            'current kurtosis': stats.kurtosis(current_clean),
            'current skewness': stats.skew(current_clean)
        }

        return features

    def calculate_entropy(self, data, bins=50):
        """
        使用自适应方法计算熵值
        :param data: array-like，输入数据
        :param bins: int或'auto'，直方图分箱数
        :return: float，熵值
        """
        data = np.array(data)

        if self.config['auto_entropy_bins']:
            # 使用Freedman-Diaconis规则自动确定bins
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            if iqr > 0:
                bin_width = 2 * iqr / (len(data) ** (1/3))
                bins = int((data.max() - data.min()) / bin_width)
                bins = max(20, min(bins, 100))  # 限制在20-100之间
            else:
                bins = 50

        # 计算熵
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        prob = hist * bin_width
        # 过滤掉零概率，避免log(0)
        prob = prob[prob > 1e-10]
        entropy = -np.sum(prob * np.log(prob + 1e-10))

        return entropy

    def assess_cycle_quality(self, cycle):
        """
        评估循环数据质量
        :param cycle: int，循环次数
        :return: dict，质量评估结果
        """
        try:
            cc_df = self.battery.get_CC_stage(cycle)
            cv_df = self.battery.get_CV_stage(cycle)

            quality = {
                'cc_points': len(cc_df),
                'cv_points': len(cv_df),
                'capacity': self.battery.dq[cycle],
                'is_valid': len(cc_df) >= 20 and len(cv_df) >= 10
            }

            # 检查容量连续性
            if cycle > 1:
                quality['capacity_change'] = abs(
                    self.battery.dq[cycle] - self.battery.dq[cycle-1]
                )
                # 容量突变检测(>50mAh为异常)
                if quality['capacity_change'] > 50:
                    quality['is_valid'] = False

            return quality
        except Exception as e:
            return {'is_valid': False, 'error': str(e)}


def process_single_battery(pkl_path, output_dir='data_features', use_robust=False):
    """
    处理单个电池文件，提取特征并保存为CSV
    :param pkl_path: str，pkl文件路径
    :param output_dir: str，输出目录
    :param use_robust: bool，是否使用鲁棒特征提取器
    """
    # 加载电池数据
    battery = Battery(pkl_path)

    # 创建特征提取器
    if use_robust:
        extractor = RobustFeatureExtractor(battery)
        print(f"使用鲁棒特征提取器 (3-sigma过滤 + 自适应熵)")
    else:
        extractor = FeatureExtractor(battery)

    # 提取特征
    features_df = extractor.extract_all_cycles()

    # 保存CSV文件
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{battery.battery_id}.csv')
    features_df.to_csv(output_path, index=False)

    print(f"特征已保存至: {output_path}")
    print(f"特征矩阵形状: {features_df.shape}")
    print("-" * 100)

    return features_df


def process_all_batteries(data_dir='our_data', output_dir='data_features', use_robust=False):
    """
    批量处理所有电池文件
    :param data_dir: str，pkl文件目录
    :param output_dir: str，输出目录
    :param use_robust: bool，是否使用鲁棒特征提取器
    """
    import glob

    # 获取所有pkl文件
    pkl_files = glob.glob(os.path.join(data_dir, '*.pkl'))
    pkl_files.sort()

    print(f"找到 {len(pkl_files)} 个电池文件")
    print("=" * 100)

    for i, pkl_path in enumerate(pkl_files, 1):
        print(f"\n[{i}/{len(pkl_files)}] 正在处理: {pkl_path}")
        try:
            process_single_battery(pkl_path, output_dir, use_robust)
        except Exception as e:
            print(f"处理失败: {str(e)}")
            continue

    print("\n" + "=" * 100)
    print("所有电池文件处理完成！")


if __name__ == '__main__':
    # 选择使用哪个版本的特征提取器
    USE_ROBUST = False  # True=鲁棒版本(推荐), False=基础版本

    # 示例1: 处理单个电池文件
    # print("示例1: 处理单个电池文件 1-1.pkl")
    # print("=" * 100)
    # df = process_single_battery('our_data/1-1.pkl', use_robust=USE_ROBUST)

    # 显示前5行数据
    # print("\n特征数据预览:")
    # print(df.head())

    # 显示统计信息
    # print("\n特征统计信息:")
    # print(df.describe())

    # 示例2: 批量处理所有电池文件
    print("\n\n示例2: 批量处理所有电池文件")
    print("=" * 100)
    if USE_ROBUST:
        print("使用鲁棒特征提取器 (推荐)")
    else:
        print("使用基础特征提取器")
    print("=" * 100)
    process_all_batteries(use_robust=USE_ROBUST)
