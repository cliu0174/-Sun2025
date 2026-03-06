"""
数据清洗工具 - 基于 PINN4SOH 论文的 3-Sigma 方法

参考: PINN4SOH/dataloader/dataloader.py
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path


def clean_3_sigma(df, verbose=True):
    """
    使用 3-Sigma 规则清洗异常值

    规则: 删除超过 mean ± 3*std 的数据点

    Args:
        df: DataFrame，包含特征和目标
        verbose: 是否打印清洗信息

    Returns:
        cleaned_df: 清洗后的 DataFrame
        stats: 清洗统计信息
    """
    original_size = len(df)

    # 1. 替换无穷大为 NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. 删除含 NaN 的行
    df = df.dropna()
    after_nan = len(df)

    # 3. 对每列应用 3-Sigma 规则
    removed_by_column = {}

    for col in df.columns:
        before = len(df)
        mean = df[col].mean()
        std = df[col].std()

        # 计算上下界
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        # 删除超过界限的行
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        removed = before - len(df)
        if removed > 0:
            removed_by_column[col] = removed

    final_size = len(df)

    # 统计信息
    stats = {
        'original_size': original_size,
        'after_nan_removal': after_nan,
        'final_size': final_size,
        'total_removed': original_size - final_size,
        'nan_removed': original_size - after_nan,
        'outliers_removed': after_nan - final_size,
        'removal_rate': (original_size - final_size) / original_size * 100,
        'removed_by_column': removed_by_column
    }

    if verbose:
        print(f"\n{'='*70}")
        print("3-Sigma 数据清洗结果")
        print(f"{'='*70}")
        print(f"原始样本数: {original_size}")
        print(f"删除 NaN: {stats['nan_removed']} ({stats['nan_removed']/original_size*100:.2f}%)")
        print(f"删除异常值: {stats['outliers_removed']} ({stats['outliers_removed']/original_size*100:.2f}%)")
        print(f"最终样本数: {final_size}")
        print(f"总删除率: {stats['removal_rate']:.2f}%")

        if removed_by_column:
            print(f"\n按列统计删除数量:")
            for col, count in removed_by_column.items():
                print(f"  {col}: {count}")
        print(f"{'='*70}\n")

    return df, stats


def load_and_clean_hust_data(data_dir='data/HUST data', apply_cleaning=True, verbose=True):
    """
    加载并清洗 HUST 数据集

    Args:
        data_dir: HUST 数据目录
        apply_cleaning: 是否应用 3-Sigma 清洗
        verbose: 是否打印详细信息

    Returns:
        all_data: 清洗后的数据字典
        cleaning_stats: 所有电池的清洗统计
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # 查找所有 CSV 文件
    csv_files = sorted(data_path.glob('*.csv'))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 CSV 文件")

    print(f"\n{'='*70}")
    print(f"加载 HUST 数据集")
    print(f"{'='*70}")
    print(f"数据目录: {data_dir}")
    print(f"找到 {len(csv_files)} 个 CSV 文件")

    all_data = {}
    cleaning_stats = {}

    for csv_file in csv_files:
        battery_name = csv_file.stem  # 文件名（不含扩展名）

        # 读取 CSV
        df = pd.read_csv(csv_file)
        original_size = len(df)

        if verbose:
            print(f"\n处理电池: {battery_name} (原始样本数: {original_size})")

        # 应用 3-Sigma 清洗
        if apply_cleaning:
            df_cleaned, stats = clean_3_sigma(df, verbose=False)
            cleaning_stats[battery_name] = stats

            if verbose:
                print(f"  清洗后: {len(df_cleaned)} 样本 "
                      f"(删除 {original_size - len(df_cleaned)} 个, "
                      f"{(original_size - len(df_cleaned))/original_size*100:.1f}%)")
        else:
            df_cleaned = df
            cleaning_stats[battery_name] = {
                'original_size': original_size,
                'final_size': len(df_cleaned),
                'total_removed': 0,
                'removal_rate': 0.0
            }

        # 分离特征和目标
        # 假设最后一列是 capacity，倒数第二列可能是 cycle_index
        if 'capacity' in df_cleaned.columns:
            features = df_cleaned.drop('capacity', axis=1)
            targets = df_cleaned['capacity'].values
        else:
            # 如果没有明确的 capacity 列，假设最后一列是目标
            features = df_cleaned.iloc[:, :-1]
            targets = df_cleaned.iloc[:, -1].values

        # 如果有 cycle_index 列，也移除
        if 'cycle_index' in features.columns:
            features = features.drop('cycle_index', axis=1)

        all_data[battery_name] = {
            'train_features': features.values,
            'train_capacity': targets
        }

    # 打印总体统计
    total_original = sum(s['original_size'] for s in cleaning_stats.values())
    total_final = sum(s['final_size'] for s in cleaning_stats.values())
    total_removed = total_original - total_final

    print(f"\n{'='*70}")
    print("总体清洗统计")
    print(f"{'='*70}")
    print(f"总电池数: {len(csv_files)}")
    print(f"原始总样本数: {total_original}")
    print(f"清洗后总样本数: {total_final}")
    print(f"总删除数: {total_removed} ({total_removed/total_original*100:.2f}%)")
    print(f"{'='*70}\n")

    return all_data, cleaning_stats


def compare_with_without_cleaning(data_dir='data/HUST data'):
    """
    对比有无数据清洗的效果

    Args:
        data_dir: HUST 数据目录
    """
    print("\n" + "="*70)
    print("数据清洗效果对比实验")
    print("="*70)

    # 加载原始数据（不清洗）
    print("\n[1] 加载原始数据（不清洗）...")
    data_original, _ = load_and_clean_hust_data(data_dir, apply_cleaning=False, verbose=False)

    # 加载清洗后数据
    print("\n[2] 加载并清洗数据（3-Sigma）...")
    data_cleaned, stats = load_and_clean_hust_data(data_dir, apply_cleaning=True, verbose=False)

    # 统计对比
    print("\n" + "="*70)
    print("数据量对比")
    print("="*70)

    for battery_name in sorted(data_original.keys()):
        original_size = len(data_original[battery_name]['train_features'])
        cleaned_size = len(data_cleaned[battery_name]['train_features'])
        removed = original_size - cleaned_size

        print(f"{battery_name:10s}: {original_size:4d} -> {cleaned_size:4d} "
              f"(删除 {removed:3d}, {removed/original_size*100:5.1f}%)")

    return data_original, data_cleaned, stats


def export_cleaning_report(cleaning_stats, output_file='data_cleaning_report.txt'):
    """
    导出清洗报告

    Args:
        cleaning_stats: 清洗统计字典
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HUST 数据集清洗报告 (3-Sigma 方法)\n")
        f.write("="*70 + "\n\n")

        # 总体统计
        total_original = sum(s['original_size'] for s in cleaning_stats.values())
        total_final = sum(s['final_size'] for s in cleaning_stats.values())
        total_removed = total_original - total_final

        f.write("总体统计:\n")
        f.write(f"  总电池数: {len(cleaning_stats)}\n")
        f.write(f"  原始总样本数: {total_original}\n")
        f.write(f"  清洗后总样本数: {total_final}\n")
        f.write(f"  总删除数: {total_removed} ({total_removed/total_original*100:.2f}%)\n\n")

        # 按电池统计
        f.write("按电池统计:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'电池名称':<15} {'原始':<8} {'清洗后':<8} {'删除':<8} {'删除率':<10}\n")
        f.write("-"*70 + "\n")

        for battery_name, stats in sorted(cleaning_stats.items()):
            f.write(f"{battery_name:<15} "
                   f"{stats['original_size']:<8} "
                   f"{stats['final_size']:<8} "
                   f"{stats['total_removed']:<8} "
                   f"{stats['removal_rate']:<10.2f}%\n")

        f.write("="*70 + "\n")

    print(f"\n清洗报告已保存到: {output_file}")


if __name__ == "__main__":
    # 示例用法

    # 方案1: 只加载并清洗数据
    print("\n方案1: 加载并清洗 HUST 数据")
    all_data, stats = load_and_clean_hust_data(
        data_dir='data/HUST data',
        apply_cleaning=True,
        verbose=True
    )

    # 导出清洗报告
    export_cleaning_report(stats, 'data_cleaning_report.txt')

    # 方案2: 对比有无清洗的效果
    # print("\n方案2: 对比有无清洗的效果")
    # data_original, data_cleaned, stats = compare_with_without_cleaning('data/HUST data')
