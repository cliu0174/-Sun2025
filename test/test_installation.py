"""
测试脚本:验证环境配置和数据加载是否正常
"""

import sys
import os

print("="*70)
print("BPINN 环境测试")
print("="*70)

# 测试依赖包导入
print("\n1. 测试依赖包...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch 未安装")
    sys.exit(1)

try:
    import numpy
    print(f"  ✓ NumPy {numpy.__version__}")
except ImportError:
    print("  ✗ NumPy 未安装")
    sys.exit(1)

try:
    import pandas
    print(f"  ✓ Pandas {pandas.__version__}")
except ImportError:
    print("  ✗ Pandas 未安装")
    sys.exit(1)

try:
    import matplotlib
    print(f"  ✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("  ✗ Matplotlib 未安装")
    sys.exit(1)

try:
    import sklearn
    print(f"  ✓ Scikit-learn {sklearn.__version__}")
except ImportError:
    print("  ✗ Scikit-learn 未安装")
    sys.exit(1)

# 检测CUDA
print("\n2. 检测计算设备...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA 可用 (设备: {torch.cuda.get_device_name(0)})")
    print(f"    CUDA 版本: {torch.version.cuda}")
else:
    print("  ○ CUDA 不可用,将使用CPU训练")

# 测试数据文件
print("\n3. 检查数据文件...")
data_files = [
    'data/B05_IC.csv',
    'data/B06_IC.csv',
    'data/B07_IC.csv'
]

all_exist = True
for file in data_files:
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} 不存在")
        all_exist = False

if not all_exist:
    print("\n  警告: 部分数据文件缺失!")

# 测试模块导入
print("\n4. 测试项目模块...")
sys.path.append('src')

try:
    from data_loader import load_battery_data, create_data_loaders
    print("  ✓ data_loader 模块")
except ImportError as e:
    print(f"  ✗ data_loader 模块: {e}")
    sys.exit(1)

try:
    from model import BPINN, BPINNLoss, SecondaryTrainingLoss
    print("  ✓ model 模块")
except ImportError as e:
    print(f"  ✗ model 模块: {e}")
    sys.exit(1)

try:
    from train import train_model, secondary_training, evaluate
    print("  ✓ train 模块")
except ImportError as e:
    print(f"  ✗ train 模块: {e}")
    sys.exit(1)

try:
    from utils import plot_training_history, plot_predictions
    print("  ✓ utils 模块")
except ImportError as e:
    print(f"  ✗ utils 模块: {e}")
    sys.exit(1)

# 测试模型实例化
print("\n5. 测试模型实例化...")
try:
    model = BPINN(input_size=6, hidden_sizes=[10, 10])
    x = torch.randn(4, 6)
    y = model(x)
    print(f"  ✓ BPINN模型创建成功")
    print(f"    输入形状: {x.shape}")
    print(f"    输出形状: {y.shape}")
    print(f"    参数数量: {sum(p.numel() for p in model.parameters())}")
except Exception as e:
    print(f"  ✗ 模型实例化失败: {e}")
    sys.exit(1)

# 测试数据加载
print("\n6. 测试数据加载...")
if all_exist:
    try:
        data = load_battery_data(data_files, train_ratio=0.6)
        print(f"  ✓ 数据加载成功")
        print(f"    训练样本: {len(data['train_features'])}")
        print(f"    测试样本: {len(data['test_features'])}")
        print(f"    特征维度: {data['train_features'].shape[1]}")
        print(f"    特征列表: {data['feature_columns']}")
    except Exception as e:
        print(f"  ✗ 数据加载失败: {e}")
        sys.exit(1)
else:
    print("  ○ 跳过(数据文件缺失)")

# 检查结果目录
print("\n7. 检查输出目录...")
if os.path.exists('results'):
    print("  ✓ results/ 目录存在")
else:
    print("  ○ results/ 目录不存在(训练时会自动创建)")

print("\n" + "="*70)
print("✓ 环境测试通过!")
print("="*70)
print("\n您可以开始训练:")
print("  python main.py")
print("\n或查看快速入门指南:")
print("  QUICKSTART.md")
print("="*70)
