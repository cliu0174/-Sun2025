"""
测试数据清洗功能集成

验证 data_loader_hust.py 中的 3-Sigma 清洗是否正常工作
"""

import sys
sys.path.append('data_loaders')

from data_loader_hust import load_single_hust_battery, load_all_hust_batteries


def test_single_battery_cleaning():
    """测试单个电池的清洗功能"""
    print("="*70)
    print("测试 1: 单个电池数据清洗")
    print("="*70)

    file_path = 'data/HUST data/1-1.csv'

    # 1. 不清洗
    print("\n[不清洗]")
    data_no_clean = load_single_hust_battery(file_path, apply_cleaning=False)
    print(f"  电池: {data_no_clean['battery_name']}")
    print(f"  训练样本: {data_no_clean['n_train']}")
    print(f"  测试样本: {data_no_clean['n_test']}")
    print(f"  总样本: {data_no_clean['n_train'] + data_no_clean['n_test']}")

    # 2. 启用清洗
    print("\n[启用3-Sigma清洗]")
    data_cleaned = load_single_hust_battery(file_path, apply_cleaning=True)
    print(f"  电池: {data_cleaned['battery_name']}")
    print(f"  训练样本: {data_cleaned['n_train']}")
    print(f"  测试样本: {data_cleaned['n_test']}")
    print(f"  总样本: {data_cleaned['n_train'] + data_cleaned['n_test']}")

    # 3. 对比
    if data_cleaned['cleaning_stats']:
        stats = data_cleaned['cleaning_stats']
        print(f"\n[清洗统计]")
        print(f"  原始样本: {stats['original_size']}")
        print(f"  清洗后样本: {stats['final_size']}")
        print(f"  删除样本: {stats['total_removed']}")
        print(f"  删除率: {stats['removal_rate']:.2f}%")

        if stats['removed_by_column']:
            print(f"\n  按列删除统计:")
            for col, count in stats['removed_by_column'].items():
                print(f"    {col}: {count}")

    print("\n[OK] 单电池测试完成!")


def test_all_batteries_cleaning():
    """测试所有电池的清洗功能"""
    print("\n" + "="*70)
    print("测试 2: 所有电池数据清洗（前5个）")
    print("="*70)

    # 1. 不清洗
    print("\n[不清洗]")
    data_no_clean = load_all_hust_batteries(
        data_dir='data/HUST data',
        max_batteries=5,
        apply_cleaning=False
    )
    total_samples_no_clean = sum(d['n_train'] + d['n_test'] for d in data_no_clean.values())
    print(f"总样本数: {total_samples_no_clean}")

    # 2. 启用清洗
    print("\n[启用3-Sigma清洗]")
    data_cleaned = load_all_hust_batteries(
        data_dir='data/HUST data',
        max_batteries=5,
        apply_cleaning=True
    )
    total_samples_cleaned = sum(d['n_train'] + d['n_test'] for d in data_cleaned.values())
    print(f"总样本数: {total_samples_cleaned}")

    # 3. 对比
    print(f"\n[对比结果]")
    print(f"  不清洗总样本: {total_samples_no_clean}")
    print(f"  清洗后总样本: {total_samples_cleaned}")
    print(f"  删除样本: {total_samples_no_clean - total_samples_cleaned}")
    print(f"  删除率: {(total_samples_no_clean - total_samples_cleaned)/total_samples_no_clean*100:.2f}%")

    print("\n[OK] 多电池测试完成!")


def test_backward_compatibility():
    """测试向后兼容性"""
    print("\n" + "="*70)
    print("测试 3: 向后兼容性")
    print("="*70)

    print("\n[验证默认行为不变（不清洗）]")
    file_path = 'data/HUST data/1-1.csv'

    # 默认参数（不指定 apply_cleaning）
    data_default = load_single_hust_battery(file_path)
    print(f"  默认加载样本数: {data_default['n_train'] + data_default['n_test']}")
    print(f"  cleaning_stats: {data_default['cleaning_stats']}")

    # 显式设置 apply_cleaning=False
    data_no_clean = load_single_hust_battery(file_path, apply_cleaning=False)
    print(f"  显式不清洗样本数: {data_no_clean['n_train'] + data_no_clean['n_test']}")

    # 验证一致
    if (data_default['n_train'] == data_no_clean['n_train'] and
        data_default['n_test'] == data_no_clean['n_test']):
        print("\n[OK] 向后兼容性测试通过！默认行为保持不变。")
    else:
        print("\n[ERROR] 向后兼容性测试失败！")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("数据清洗集成测试")
    print("="*70)

    try:
        # 测试1: 单个电池
        test_single_battery_cleaning()

        # 测试2: 多个电池
        test_all_batteries_cleaning()

        # 测试3: 向后兼容性
        test_backward_compatibility()

        print("\n" + "="*70)
        print("所有测试完成！")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
