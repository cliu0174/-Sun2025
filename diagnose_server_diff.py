"""
诊断服务器和本地结果差异的脚本

问题描述：
- 本地：compare_models.py 和 train_cross_battery.py 结果一致 ✓
- 服务器：compare_models.py 和 train_cross_battery.py 结果不一致 ✗

可能原因：
1. Python缓存文件(.pyc)导致旧代码被执行
2. 文件传输时编码/换行符问题
3. DataLoader的worker进程行为差异
4. 导入时的全局状态差异
5. 随机数生成器在多次调用时的状态
"""

import os
import sys
import hashlib


def get_file_hash(filepath):
    """计算文件的MD5哈希"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def check_file_integrity():
    """检查关键文件的完整性"""
    print("=" * 70)
    print("文件完整性检查")
    print("=" * 70)

    files_to_check = [
        'compare_models.py',
        'train_cross_battery.py',
        'configs/models/cnn_lstm_config.json',
        'models/model_factory.py',
        'models/cnn_lstm.py',
        'data_loaders/data_loader_hust.py',
    ]

    print("\n文件MD5哈希值（用于验证文件传输完整性）:")
    for filepath in files_to_check:
        file_hash = get_file_hash(filepath)
        if file_hash:
            print(f"  {filepath}: {file_hash}")
        else:
            print(f"  {filepath}: 文件不存在")

    print("\n说明：在本地和服务器上运行此脚本，对比MD5值")
    print("如果MD5不同，说明文件内容有差异（可能是换行符或编码问题）")


def check_line_endings():
    """检查换行符类型"""
    print("\n" + "=" * 70)
    print("换行符检查")
    print("=" * 70)

    files = ['compare_models.py', 'train_cross_battery.py']

    for filepath in files:
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'rb') as f:
            content = f.read()

        crlf_count = content.count(b'\r\n')
        lf_count = content.count(b'\n') - crlf_count
        cr_count = content.count(b'\r') - crlf_count

        print(f"\n{filepath}:")
        print(f"  CRLF (\\r\\n): {crlf_count} (Windows)")
        print(f"  LF (\\n): {lf_count} (Unix/Linux/Mac)")
        print(f"  CR (\\r): {cr_count} (Old Mac)")

        if crlf_count > 0 and lf_count > 0:
            print(f"  ⚠️ 警告: 混合换行符！")


def check_cache_files():
    """检查Python缓存文件"""
    print("\n" + "=" * 70)
    print("Python缓存检查")
    print("=" * 70)

    import glob

    # 查找所有 .pyc 文件
    pyc_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                pyc_files.append(os.path.join(root, file))

    print(f"\n找到 {len(pyc_files)} 个 .pyc 缓存文件")

    if pyc_files:
        print("\n前10个缓存文件:")
        for pyc in pyc_files[:10]:
            print(f"  {pyc}")

        print("\n⚠️ 建议清理缓存:")
        print("  方法1: find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null")
        print("  方法2: find . -name '*.pyc' -delete")
        print("  方法3: python -Bc \"import compileall; compileall.compile_dir('.', force=True)\"")


def test_import_consistency():
    """测试导入的一致性"""
    print("\n" + "=" * 70)
    print("导入一致性测试")
    print("=" * 70)

    try:
        # 清除已导入的模块
        if 'train_cross_battery' in sys.modules:
            del sys.modules['train_cross_battery']

        # 导入函数
        from train_cross_battery import train_cross_battery_model, set_seed

        # 检查函数位置
        print(f"\ntrain_cross_battery_model 位置: {train_cross_battery_model.__module__}")
        print(f"set_seed 位置: {set_seed.__module__}")

        # 检查默认参数
        import inspect
        sig = inspect.signature(train_cross_battery_model)
        print(f"\ntrain_cross_battery_model 默认参数:")
        for param_name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                print(f"  {param_name} = {param.default}")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


def check_dataloader_workers():
    """检查DataLoader worker设置"""
    print("\n" + "=" * 70)
    print("DataLoader Worker配置检查")
    print("=" * 70)

    try:
        # 读取 train_cross_battery.py 查找 num_workers
        with open('train_cross_battery.py', 'r', encoding='utf-8') as f:
            content = f.read()

        import re
        workers_matches = re.findall(r'num_workers\s*=\s*(\d+)', content)

        print(f"\n在 train_cross_battery.py 中找到的 num_workers 设置:")
        for i, worker_val in enumerate(set(workers_matches), 1):
            count = workers_matches.count(worker_val)
            print(f"  num_workers={worker_val}: 出现 {count} 次")

        if '0' not in workers_matches:
            print("\n⚠️ 警告: DataLoader 没有设置 num_workers=0")
            print("   多进程加载可能导致不确定性，建议设为0")

    except Exception as e:
        print(f"\n无法检查: {e}")


def check_set_seed_calls():
    """检查 set_seed 调用"""
    print("\n" + "=" * 70)
    print("随机种子设置检查")
    print("=" * 70)

    files_to_check = {
        'compare_models.py': ['set_seed', 'random.seed', 'np.random.seed', 'torch.manual_seed'],
        'train_cross_battery.py': ['set_seed', 'random.seed', 'np.random.seed', 'torch.manual_seed'],
    }

    for filepath, keywords in files_to_check.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            print(f"\n{filepath}:")
            for keyword in keywords:
                matches = [i+1 for i, line in enumerate(lines) if keyword in line and not line.strip().startswith('#')]
                if matches:
                    print(f"  {keyword}: 第 {matches} 行")

        except Exception as e:
            print(f"\n{filepath}: 无法读取 - {e}")


def check_cudnn_settings():
    """检查cuDNN设置"""
    print("\n" + "=" * 70)
    print("cuDNN确定性设置检查")
    print("=" * 70)

    try:
        import torch

        print(f"\ntorch.backends.cudnn.deterministic: {torch.backends.cudnn.deterministic}")
        print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

        if not torch.backends.cudnn.deterministic:
            print("\n⚠️ 警告: cudnn.deterministic 未设置为 True")
            print("   这可能导致结果不可复现")

        if torch.backends.cudnn.benchmark:
            print("\n⚠️ 警告: cudnn.benchmark 设置为 True")
            print("   这可能导致结果不可复现")

    except Exception as e:
        print(f"\n无法检查: {e}")


def suggest_fixes():
    """建议修复方案"""
    print("\n" + "=" * 70)
    print("建议的修复方案")
    print("=" * 70)

    suggestions = [
        "1. 清理Python缓存:",
        "   cd /path/to/project",
        "   find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null",
        "   find . -name '*.pyc' -delete",
        "",
        "2. 转换换行符为Unix格式（如果使用Windows开发）:",
        "   dos2unix *.py",
        "   或使用 sed: sed -i 's/\\r$//' *.py",
        "",
        "3. 确保DataLoader使用 num_workers=0:",
        "   在 train_cross_battery.py 的所有 DataLoader 调用中设置 num_workers=0",
        "",
        "4. 在 compare_models.py 中每次调用前重置随机种子:",
        "   在 train_cross_battery_model() 调用前显式调用 set_seed(seed)",
        "",
        "5. 检查是否有全局状态:",
        "   确保 train_cross_battery.py 中没有模块级别的全局变量保存状态",
        "",
        "6. 使用完全相同的Python环境:",
        "   pip freeze > requirements.txt (本地)",
        "   pip install -r requirements.txt (服务器)",
    ]

    for line in suggestions:
        print(line)


def main():
    print("\n" + "=" * 70)
    print("服务器结果差异诊断脚本")
    print("=" * 70)
    print("请在本地和服务器上都运行此脚本，对比输出找出差异")
    print()

    check_file_integrity()
    check_line_endings()
    check_cache_files()
    test_import_consistency()
    check_dataloader_workers()
    check_set_seed_calls()
    check_cudnn_settings()
    suggest_fixes()

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
