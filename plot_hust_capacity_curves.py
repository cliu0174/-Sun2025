"""
绘制HUST数据集中电池容量随循环次数变化的曲线。

使用方法:
    # 绘制单个电池
    python plot_hust_capacity_curves.py --battery 1-1
    
    # 绘制多个电池
    python plot_hust_capacity_curves.py --batteries 1-1 1-2 1-3
    
    # 绘制所有B电池组
    python plot_hust_capacity_curves.py --battery-group B
    
    # 自定义保存路径
    python plot_hust_capacity_curves.py --battery 1-1 --save-path results/capacity_curves.png
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_loaders import load_single_hust_battery
from src.utils import ensure_dir


def plot_single_capacity_curve(battery_name, data_path, figsize=(10, 6)):
    """
    绘制单个电池的容量衰减曲线。
    
    Args:
        battery_name: 电池名称 (e.g., '1-1')
        data_path: 数据文件路径
        figsize: 图表大小
        
    Returns:
        capacity values (numpy array)
    """
    # 读取CSV文件
    df = pd.read_csv(data_path)
    
    # 提取容量列
    capacity = df['capacity'].values
    cycles = np.arange(1, len(capacity) + 1)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制容量曲线
    ax.plot(cycles, capacity, 'b-', linewidth=2, label='Capacity')
    
    # 设置标签和标题
    ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capacity (Ah)', fontsize=12, fontweight='bold')
    ax.set_title(f'Battery {battery_name} - Capacity Degradation', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(fontsize=10)
    
    # 添加统计信息
    initial_capacity = capacity[0]
    final_capacity = capacity[-1]
    capacity_loss = initial_capacity - final_capacity
    capacity_loss_pct = (capacity_loss / initial_capacity) * 100
    
    stats_text = (
        f'Initial Capacity: {initial_capacity:.4f} Ah\n'
        f'Final Capacity: {final_capacity:.4f} Ah\n'
        f'Total Loss: {capacity_loss:.4f} Ah ({capacity_loss_pct:.2f}%)\n'
        f'Total Cycles: {len(capacity)}'
    )
    
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, family='monospace')
    
    plt.tight_layout()
    
    return capacity, cycles


def plot_multiple_capacity_curves(battery_names, data_dir='data/HUST data', 
                                   figsize=(14, 8)):
    """
    绘制多个电池的容量衰减曲线对比。
    
    Args:
        battery_names: 电池名称列表 (e.g., ['1-1', '1-2', '1-3'])
        data_dir: 数据目录
        figsize: 图表大小
        
    Returns:
        (fig, ax, data_dict)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    data_dict = {}
    colors = plt.cm.tab20(np.linspace(0, 1, len(battery_names)))
    
    for idx, battery_name in enumerate(battery_names):
        data_path = os.path.join(data_dir, f'{battery_name}.csv')
        
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found for battery {battery_name}: {data_path}")
            continue
        
        # 读取数据
        df = pd.read_csv(data_path)
        capacity = df['capacity'].values
        cycles = np.arange(1, len(capacity) + 1)
        
        # 保存数据
        data_dict[battery_name] = {
            'cycles': cycles,
            'capacity': capacity,
            'initial': capacity[0],
            'final': capacity[-1],
            'loss': capacity[0] - capacity[-1]
        }
        
        # 绘制曲线
        ax.plot(cycles, capacity, linewidth=2, label=f'Battery {battery_name}',
                color=colors[idx])
    
    # 设置标签和标题
    ax.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capacity (Ah)', fontsize=12, fontweight='bold')
    ax.set_title(f'Capacity Degradation Comparison - {len(battery_names)} Batteries',
                 fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    
    return fig, ax, data_dict


def plot_capacity_statistics(battery_names, data_dir='data/HUST data',
                              figsize=(12, 6)):
    """
    绘制电池容量损失的统计信息。
    
    Args:
        battery_names: 电池名称列表
        data_dir: 数据目录
        figsize: 图表大小
        
    Returns:
        统计数据字典
    """
    stats = {
        'battery': [],
        'initial_capacity': [],
        'final_capacity': [],
        'capacity_loss': [],
        'capacity_loss_pct': [],
        'total_cycles': []
    }
    
    for battery_name in battery_names:
        data_path = os.path.join(data_dir, f'{battery_name}.csv')
        
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found for battery {battery_name}")
            continue
        
        df = pd.read_csv(data_path)
        capacity = df['capacity'].values
        
        initial = capacity[0]
        final = capacity[-1]
        loss = initial - final
        loss_pct = (loss / initial) * 100
        
        stats['battery'].append(battery_name)
        stats['initial_capacity'].append(initial)
        stats['final_capacity'].append(final)
        stats['capacity_loss'].append(loss)
        stats['capacity_loss_pct'].append(loss_pct)
        stats['total_cycles'].append(len(capacity))
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. 初始容量对比
    axes[0, 0].bar(stats['battery'], stats['initial_capacity'], color='green', alpha=0.7)
    axes[0, 0].set_ylabel('Capacity (Ah)', fontsize=10)
    axes[0, 0].set_title('Initial Capacity', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. 最终容量对比
    axes[0, 1].bar(stats['battery'], stats['final_capacity'], color='red', alpha=0.7)
    axes[0, 1].set_ylabel('Capacity (Ah)', fontsize=10)
    axes[0, 1].set_title('Final Capacity', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. 容量损失（绝对值）
    axes[1, 0].bar(stats['battery'], stats['capacity_loss'], color='orange', alpha=0.7)
    axes[1, 0].set_ylabel('Capacity Loss (Ah)', fontsize=10)
    axes[1, 0].set_title('Total Capacity Loss', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. 容量损失百分比
    axes[1, 1].bar(stats['battery'], stats['capacity_loss_pct'], color='purple', alpha=0.7)
    axes[1, 1].set_ylabel('Loss Percentage (%)', fontsize=10)
    axes[1, 1].set_title('Capacity Loss Percentage', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 旋转x轴标签
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig, stats


def get_battery_list(battery_group=None, data_dir='data/HUST data'):
    """
    获取电池列表。
    
    Args:
        battery_group: 电池组 ('A', 'B', 'C', 等) 或 None (所有电池)
        data_dir: 数据目录
        
    Returns:
        电池名称列表
    """
    batteries = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                battery_name = file.replace('.csv', '')
                
                if battery_group is None:
                    batteries.append(battery_name)
                elif battery_name.startswith(battery_group):
                    batteries.append(battery_name)
    
    return sorted(batteries)


def main():
    parser = argparse.ArgumentParser(
        description='Plot capacity degradation curves for HUST battery dataset'
    )
    
    parser.add_argument('--battery', type=str, default=None,
                        help='Single battery to plot (e.g., "1-1")')
    parser.add_argument('--batteries', nargs='+', default=None,
                        help='Multiple batteries to plot (e.g., "1-1 1-2 1-3")')
    parser.add_argument('--battery-group', type=str, default=None,
                        help='Battery group to plot (e.g., "1", "2", "B")')
    parser.add_argument('--data-dir', type=str, default='data/HUST data',
                        help='Data directory path')
    parser.add_argument('--save-dir', type=str, default='results/capacity_curves',
                        help='Directory to save plots')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Custom save path for single battery plot')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures')
    
    args = parser.parse_args()
    
    # 确定要绘制的电池
    if args.battery:
        batteries = [args.battery]
    elif args.batteries:
        batteries = args.batteries
    elif args.battery_group:
        batteries = get_battery_list(args.battery_group, args.data_dir)
    else:
        # 默认绘制前10个电池
        all_batteries = get_battery_list(None, args.data_dir)
        batteries = all_batteries[:10]
        print(f"No battery specified. Plotting first 10 batteries: {batteries}")
    
    print(f"\nPlotting capacity curves for {len(batteries)} battery(ies)...")
    print(f"Batteries: {batteries}")
    
    # 创建保存目录
    ensure_dir(args.save_dir)
    
    # 1. 单个电池模式
    if args.battery:
        print(f"\n{'='*70}")
        print(f"Single Battery Mode: {args.battery}")
        print(f"{'='*70}")
        
        data_path = os.path.join(args.data_dir, f'{args.battery}.csv')
        
        if not os.path.exists(data_path):
            print(f"Error: Data file not found: {data_path}")
            return
        
        capacity, cycles = plot_single_capacity_curve(args.battery, data_path)
        
        # 保存图表
        if args.save_path:
            save_path = args.save_path
        else:
            safe_name = args.battery.replace('-', '_').replace('/', '_')
            save_path = os.path.join(args.save_dir, f'{safe_name}_capacity_curve.png')
        
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
        print(f"\n✓ Single battery plot saved: {save_path}")
        
        # 显示统计信息
        print(f"\n电池 {args.battery} 容量统计:")
        print(f"  初始容量: {capacity[0]:.6f} Ah")
        print(f"  最终容量: {capacity[-1]:.6f} Ah")
        print(f"  容量损失: {capacity[0]-capacity[-1]:.6f} Ah ({((capacity[0]-capacity[-1])/capacity[0])*100:.2f}%)")
        print(f"  总循环数: {len(capacity)}")
    
    # 2. 多电池对比模式
    else:
        print(f"\n{'='*70}")
        print(f"Multiple Batteries Mode: {len(batteries)} batteries")
        print(f"{'='*70}")
        
        # 绘制容量曲线对比
        fig, ax, data_dict = plot_multiple_capacity_curves(batteries, args.data_dir)
        
        save_path = os.path.join(args.save_dir, 'capacity_comparison.png')
        plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved: {save_path}")
        
        # 绘制统计信息
        fig, stats = plot_capacity_statistics(batteries, args.data_dir)
        
        save_path = os.path.join(args.save_dir, 'capacity_statistics.png')
        plt.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
        print(f"✓ Statistics plot saved: {save_path}")
        
        # 打印统计信息
        print(f"\n{'电池':<10} {'初始容量':<12} {'最终容量':<12} {'损失(Ah)':<12} {'损失(%)':<10} {'循环数':<8}")
        print("-" * 70)
        for i, battery in enumerate(stats['battery']):
            print(f"{battery:<10} {stats['initial_capacity'][i]:<12.6f} "
                  f"{stats['final_capacity'][i]:<12.6f} {stats['capacity_loss'][i]:<12.6f} "
                  f"{stats['capacity_loss_pct'][i]:<10.2f} {stats['total_cycles'][i]:<8}")
    
    # 显示图表
    if not args.no_show:
        plt.show()
    
    print(f"\n✓ All plots saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
