"""
快速检查脚本：验证本地和服务器环境差异

运行此脚本只需几秒钟，可以快速发现问题
"""

import os
import sys
import hashlib

def quick_check():
    print("=" * 70)
    print("快速环境检查")
    print("=" * 70)

    # 1. 检查关键文件MD5
    print("\n1. 关键文件MD5:")
    key_files = ['compare_models.py', 'train_cross_battery.py', 'configs/models/cnn_lstm_config.json']
    for f in key_files:
        if os.path.exists(f):
            with open(f, 'rb') as file:
                md5 = hashlib.md5(file.read()).hexdigest()
            print(f"   {f}: {md5}")
        else:
            print(f"   {f}: 文件不存在")

    # 2. 检查Python缓存
    print("\n2. Python缓存检查:")
    pycache_count = sum(1 for root, dirs, files in os.walk('.')
                        for d in dirs if d == '__pycache__')
    pyc_count = sum(1 for root, dirs, files in os.walk('.')
                    for f in files if f.endswith('.pyc'))
    print(f"   __pycache__ 目录数: {pycache_count}")
    print(f"   .pyc 文件数: {pyc_count}")
    if pycache_count > 0 or pyc_count > 0:
        print("   ⚠️ 建议清理缓存: find . -name '__pycache__' -o -name '*.pyc' | xargs rm -rf")

    # 3. 检查换行符
    print("\n3. 换行符检查 (compare_models.py):")
    try:
        with open('compare_models.py', 'rb') as f:
            content = f.read()
        crlf = content.count(b'\r\n')
        lf_only = content.count(b'\n') - crlf
        print(f"   CRLF (Windows): {crlf}")
        print(f"   LF (Unix): {lf_only}")
        if crlf > 0 and lf_only > 0:
            print("   ⚠️ 混合换行符！建议: dos2unix *.py")
    except:
        print("   无法读取")

    # 4. 检查torch设置
    print("\n4. PyTorch环境:")
    try:
        import torch
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   cudnn.deterministic: {torch.backends.cudnn.deterministic}")
        print(f"   cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    except ImportError:
        print("   PyTorch未安装")

    # 5. 模拟电池划分
    print("\n5. 电池划分测试 (seed=42):")
    try:
        import random
        data_dir = 'data/HUST data'
        if os.path.exists(data_dir):
            csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
            battery_names = [f.replace('.csv', '') for f in csv_files]

            random.seed(42)
            shuffled = battery_names.copy()
            random.shuffle(shuffled)

            n_train = int(len(shuffled) * 0.6)
            train_batteries = shuffled[:n_train]

            print(f"   总电池数: {len(battery_names)}")
            print(f"   训练集前3个: {train_batteries[:3]}")
            print(f"   训练集哈希: {hash(tuple(train_batteries))}")
        else:
            print("   数据目录不存在")
    except Exception as e:
        print(f"   错误: {e}")

    print("\n" + "=" * 70)
    print("检查完成")
    print("=" * 70)
    print("\n对比本地和服务器的输出，重点关注:")
    print("1. MD5值是否完全相同")
    print("2. 是否有Python缓存文件")
    print("3. 换行符是否一致")
    print("4. PyTorch版本是否一致")
    print("5. 训练集前3个电池和哈希是否一致")

if __name__ == '__main__':
    quick_check()
