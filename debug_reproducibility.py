"""
复现性诊断脚本：排查为什么服务器和本地结果不一致

使用方法：
1. 在本地运行: python debug_reproducibility.py
2. 在服务器运行: python debug_reproducibility.py
3. 对比两边输出的所有信息，找出差异
"""

import os
import sys
import random
import numpy as np
import torch
import json
from pathlib import Path

def check_environment():
    """检查环境信息"""
    print("=" * 70)
    print("环境信息检查")
    print("=" * 70)

    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"脚本路径: {os.path.abspath(__file__)}")

    print(f"\nPyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")

    print(f"\nNumPy版本: {np.__version__}")

    # 检查关键库
    try:
        import pandas as pd
        print(f"Pandas版本: {pd.__version__}")
    except ImportError:
        print("Pandas: 未安装")

    try:
        import sklearn
        print(f"Scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn: 未安装")


def check_random_state():
    """检查随机数生成器状态"""
    print("\n" + "=" * 70)
    print("随机数生成器测试")
    print("=" * 70)

    # 设置种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 生成随机数
    print(f"\n使用种子 {seed} 生成的随机数:")
    print(f"random.random(): {random.random()}")
    print(f"np.random.rand(): {np.random.rand()}")
    print(f"torch.rand(1): {torch.rand(1).item()}")

    # 测试 shuffle
    test_list = list(range(10))
    random.seed(seed)
    random.shuffle(test_list)
    print(f"\nrandom.shuffle([0,1,2,...,9]): {test_list}")


def check_data_loading():
    """检查数据加载"""
    print("\n" + "=" * 70)
    print("数据加载检查")
    print("=" * 70)

    data_dir = 'data/HUST data'
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        return

    # 列出所有CSV文件
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    print(f"\n找到 {len(csv_files)} 个CSV文件")
    print(f"前5个文件: {csv_files[:5]}")
    print(f"后5个文件: {csv_files[-5:]}")

    # 检查文件排序是否一致
    print(f"\n文件排序哈希: {hash(tuple(csv_files))}")


def check_config_file():
    """检查配置文件"""
    print("\n" + "=" * 70)
    print("配置文件检查")
    print("=" * 70)

    config_path = 'configs/models/cnn_lstm_config.json'
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"\n配置文件路径: {os.path.abspath(config_path)}")
    print(f"配置文件大小: {os.path.getsize(config_path)} 字节")

    # 检查关键配置
    print(f"\n关键配置:")
    print(f"  window_size: {config['data'].get('window_size')}")
    print(f"  batch_size: {config['training'].get('batch_size')}")
    print(f"  learning_rate: {config['training'].get('learning_rate')}")
    print(f"  num_epochs: {config['training'].get('num_epochs')}")
    print(f"  physics_enabled: {config['physics_constraints'].get('enabled')}")
    print(f"  monotonic_weight: {config['physics_constraints'].get('monotonic_weight')}")

    # 计算配置的哈希值（用于检测差异）
    config_str = json.dumps(config, sort_keys=True)
    print(f"\n配置内容哈希: {hash(config_str)}")


def check_battery_split():
    """检查电池划分的一致性"""
    print("\n" + "=" * 70)
    print("电池划分一致性检查")
    print("=" * 70)

    data_dir = 'data/HUST data'
    if not os.path.exists(data_dir):
        print(f"警告: 数据目录不存在: {data_dir}")
        return

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    battery_names = [f.replace('.csv', '') for f in csv_files]

    # 模拟 split_batteries 函数
    seed = 42
    train_ratio = 0.6
    val_ratio = 0.2

    random.seed(seed)
    np.random.seed(seed)

    shuffled_batteries = battery_names.copy()
    random.shuffle(shuffled_batteries)

    n_total = len(shuffled_batteries)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_batteries = shuffled_batteries[:n_train]
    val_batteries = shuffled_batteries[n_train:n_train + n_val]
    test_batteries = shuffled_batteries[n_train + n_val:]

    print(f"\n总电池数: {n_total}")
    print(f"训练集: {len(train_batteries)} 个")
    print(f"验证集: {len(val_batteries)} 个")
    print(f"测试集: {len(test_batteries)} 个")

    print(f"\n训练集前5个: {train_batteries[:5]}")
    print(f"验证集前5个: {val_batteries[:5]}")
    print(f"测试集前5个: {test_batteries[:5]}")

    # 计算哈希
    print(f"\n训练集哈希: {hash(tuple(train_batteries))}")
    print(f"验证集哈希: {hash(tuple(val_batteries))}")
    print(f"测试集哈希: {hash(tuple(test_batteries))}")


def check_model_creation():
    """检查模型创建的一致性"""
    print("\n" + "=" * 70)
    print("模型创建一致性检查")
    print("=" * 70)

    try:
        from models import ModelFactory, ConfigLoader

        # 加载配置
        config = ConfigLoader.load_model_config('cnn_lstm')

        # 设置随机种子
        seed = 42
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # 创建模型
        model = ModelFactory.create_model('cnn_lstm', input_size=6, config=config)

        # 计算模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n模型参数总数: {total_params}")
        print(f"可训练参数: {trainable_params}")

        # 获取第一层的权重统计（用于检测初始化差异）
        first_param = next(model.parameters())
        print(f"\n第一层参数形状: {first_param.shape}")
        print(f"第一层参数均值: {first_param.mean().item():.6f}")
        print(f"第一层参数标准差: {first_param.std().item():.6f}")
        print(f"第一层前5个参数: {first_param.flatten()[:5].tolist()}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def check_dataloader_determinism():
    """检查DataLoader的确定性"""
    print("\n" + "=" * 70)
    print("DataLoader确定性检查")
    print("=" * 70)

    # 创建简单数据集
    from torch.utils.data import TensorDataset, DataLoader

    seed = 42
    torch.manual_seed(seed)

    # 创建测试数据
    data = torch.randn(100, 10)
    labels = torch.randn(100)
    dataset = TensorDataset(data, labels)

    # 测试不同 num_workers 设置
    for num_workers in [0, 2]:
        torch.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=num_workers)

        first_batch = next(iter(loader))
        print(f"\nnum_workers={num_workers}:")
        print(f"  第一批数据均值: {first_batch[0].mean().item():.6f}")
        print(f"  第一批数据前3个样本的第一个特征: {first_batch[0][:3, 0].tolist()}")


def check_python_cache():
    """检查Python缓存"""
    print("\n" + "=" * 70)
    print("Python缓存检查")
    print("=" * 70)

    # 查找 .pyc 文件
    pyc_files = list(Path('.').rglob('*.pyc'))
    print(f"\n找到 {len(pyc_files)} 个 .pyc 文件")

    if pyc_files:
        print("建议清理缓存:")
        print("  Unix/Linux/Mac: find . -type d -name __pycache__ -exec rm -rf {} +")
        print("  Windows: for /d /r . %d in (__pycache__) do @if exist \"%d\" rd /s /q \"%d\"")


def main():
    print("\n" + "=" * 70)
    print("复现性诊断脚本")
    print("=" * 70)
    print("此脚本将检查可能导致结果不一致的因素")
    print("请在本地和服务器上都运行此脚本，然后对比输出")
    print()

    check_environment()
    check_random_state()
    check_data_loading()
    check_config_file()
    check_battery_split()
    check_model_creation()
    check_dataloader_determinism()
    check_python_cache()

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
    print("\n建议:")
    print("1. 对比本地和服务器的所有输出，特别注意哈希值是否一致")
    print("2. 如果哈希值不一致，说明数据/配置/随机性有差异")
    print("3. 检查 PyTorch 和 CUDA 版本是否一致")
    print("4. 清理 Python 缓存后重试")
    print("5. 确保 train_cross_battery.py 中 set_seed() 在每次训练前都被调用")
    print("6. 检查 DataLoader 的 num_workers 参数（建议设为0以确保完全确定性）")


if __name__ == '__main__':
    main()
