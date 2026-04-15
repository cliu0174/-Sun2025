"""
分析电池早期容量回升现象
统计有多少电池出现容量回升，峰值循环范围，并绘制放大示意图
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import os

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150

def load_all_batteries(data_dir='our_data'):
    """加载所有电池数据"""
    batteries = {}
    data_path = Path(data_dir)

    for pkl_file in data_path.glob('*.pkl'):
        battery_id = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        batteries[battery_id] = data[battery_id]

    return batteries

def analyze_capacity_rise(battery_data, early_cycle_limit=500):
    """
    分析单个电池是否存在早期容量回升

    Args:
        battery_data: 电池数据字典
        early_cycle_limit: 早期循环的定义范围

    Returns:
        dict: 分析结果
    """
    cycles = sorted(battery_data['dq'].keys())
    capacities = np.array([battery_data['dq'][c] for c in cycles])

    # 计算SOH
    initial_capacity = capacities[0]
    soh = capacities / initial_capacity * 100

    # 只分析早期循环
    early_mask = np.array(cycles) <= early_cycle_limit
    early_cycles = np.array(cycles)[early_mask]
    early_soh = soh[early_mask]

    if len(early_soh) < 10:
        return None

    # 检测容量回升：找到早期的最大值
    max_idx = np.argmax(early_soh)
    max_cycle = early_cycles[max_idx]
    max_soh = early_soh[max_idx]

    # 判断是否存在回升（最大值不在第一个点，且比初始值高）
    has_rise = max_idx > 0 and max_soh > soh[0] + 0.1  # 至少0.1%的回升

    # 计算回升幅度
    rise_amplitude = max_soh - soh[0] if has_rise else 0

    return {
        'has_rise': has_rise,
        'peak_cycle': int(max_cycle) if has_rise else None,
        'peak_soh': float(max_soh) if has_rise else None,
        'initial_soh': float(soh[0]),
        'rise_amplitude': float(rise_amplitude),
        'cycles': cycles,
        'soh': soh.tolist(),
        'early_cycles': early_cycles.tolist(),
        'early_soh': early_soh.tolist()
    }

def main():
    print("=" * 70)
    print("电池早期容量回升现象分析")
    print("=" * 70)

    # 加载所有电池
    batteries = load_all_batteries()
    print(f"\n加载了 {len(batteries)} 个电池")

    # 分析每个电池
    results = {}
    rise_batteries = []

    for battery_id in sorted(batteries.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1]))):
        result = analyze_capacity_rise(batteries[battery_id])
        if result:
            results[battery_id] = result
            if result['has_rise']:
                rise_batteries.append({
                    'id': battery_id,
                    'peak_cycle': result['peak_cycle'],
                    'peak_soh': result['peak_soh'],
                    'initial_soh': result['initial_soh'],
                    'rise_amplitude': result['rise_amplitude']
                })

    # 统计结果
    print(f"\n分析完成的电池数: {len(results)}")
    print(f"存在早期容量回升的电池数: {len(rise_batteries)}")
    print(f"占比: {len(rise_batteries)/len(results)*100:.1f}%")

    if rise_batteries:
        # 峰值循环统计
        peak_cycles = [b['peak_cycle'] for b in rise_batteries]
        rise_amplitudes = [b['rise_amplitude'] for b in rise_batteries]

        print(f"\n峰值循环范围: {min(peak_cycles)} ~ {max(peak_cycles)}")
        print(f"峰值循环均值: {np.mean(peak_cycles):.1f}")
        print(f"峰值循环中位数: {np.median(peak_cycles):.1f}")
        print(f"\n回升幅度范围: {min(rise_amplitudes):.2f}% ~ {max(rise_amplitudes):.2f}%")
        print(f"回升幅度均值: {np.mean(rise_amplitudes):.2f}%")

        print("\n" + "-" * 70)
        print("存在容量回升的电池详情:")
        print("-" * 70)
        print(f"{'电池ID':<10} {'峰值循环':<12} {'初始SOH':<12} {'峰值SOH':<12} {'回升幅度':<10}")
        print("-" * 70)

        for b in sorted(rise_batteries, key=lambda x: x['rise_amplitude'], reverse=True):
            print(f"{b['id']:<10} {b['peak_cycle']:<12} {b['initial_soh']:<12.2f} {b['peak_soh']:<12.2f} {b['rise_amplitude']:<10.2f}%")

    # ==================== 绘图 ====================

    # 选择几个典型的容量回升电池进行展示
    if rise_batteries:
        # 按回升幅度排序，选择前6个
        top_rise = sorted(rise_batteries, key=lambda x: x['rise_amplitude'], reverse=True)[:6]

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for idx, battery_info in enumerate(top_rise):
            ax = axes[idx]
            battery_id = battery_info['id']
            result = results[battery_id]

            cycles = np.array(result['cycles'])
            soh = np.array(result['soh'])

            # 绘制完整曲线（灰色）
            ax.plot(cycles, soh, 'gray', alpha=0.3, linewidth=1, label='Full curve')

            # 绘制早期部分（放大区域）
            early_mask = cycles <= 500
            ax.plot(cycles[early_mask], soh[early_mask], 'b-', linewidth=2, label='Early cycles')

            # 标注峰值点
            peak_cycle = battery_info['peak_cycle']
            peak_soh = battery_info['peak_soh']
            ax.scatter([peak_cycle], [peak_soh], color='red', s=100, zorder=5,
                      edgecolor='white', linewidth=2, label=f'Peak @ cycle {peak_cycle}')

            # 标注初始点
            ax.scatter([cycles[0]], [soh[0]], color='green', s=80, zorder=5,
                      edgecolor='white', linewidth=2, label='Initial')

            # 绘制回升幅度标注
            ax.annotate('', xy=(peak_cycle, peak_soh), xytext=(peak_cycle, soh[0]),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
            ax.text(peak_cycle + 20, (peak_soh + soh[0])/2,
                   f'+{battery_info["rise_amplitude"]:.2f}%',
                   fontsize=9, color='red', fontweight='bold')

            ax.set_xlim([0, 500])
            ax.set_ylim([min(soh[early_mask])*0.995, max(soh[early_mask])*1.005])
            ax.set_xlabel('Cycle Number', fontsize=10)
            ax.set_ylabel('SOH (%)', fontsize=10)
            ax.set_title(f'Battery {battery_id}\nRise: +{battery_info["rise_amplitude"]:.2f}%',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='lower left', fontsize=8)

        plt.suptitle('Early Capacity Rise Phenomenon in Li-ion Batteries\n(Zoomed view: Cycles 0-500)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存
        plt.savefig('capacity_rise_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"\n图片已保存: capacity_rise_analysis.png")
        plt.close()

        # ==================== 图1：多条曲线叠加对比 ====================
        fig2, ax1 = plt.subplots(figsize=(10, 6))

        for idx, battery_info in enumerate(top_rise[:8]):
            battery_id = battery_info['id']
            result = results[battery_id]
            cycles = np.array(result['cycles'])
            soh = np.array(result['soh'])
            early_mask = cycles <= 400

            ax1.plot(cycles[early_mask], soh[early_mask], linewidth=1.8, alpha=0.85,
                    label=f'{battery_id} (+{battery_info["rise_amplitude"]:.1f}%)')

        ax1.set_xlabel('Cycle Number', fontsize=12)
        ax1.set_ylabel('SOH (%)', fontsize=12)
        ax1.set_title('Early Capacity Rise: Multiple Batteries Comparison',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='lower left', fontsize=10, ncol=2)
        ax1.set_xlim([0, 400])
        ax1.set_yticklabels([])  # 隐藏Y轴刻度标签

        plt.tight_layout()
        plt.savefig('capacity_rise_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"图片已保存: capacity_rise_comparison.png")
        plt.close()

        # ==================== 图2：峰值循环分布直方图 ====================
        fig3, ax2 = plt.subplots(figsize=(8, 6))

        ax2.hist(peak_cycles, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
        ax2.axvline(np.mean(peak_cycles), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(peak_cycles):.0f}')
        ax2.axvline(np.median(peak_cycles), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(peak_cycles):.0f}')
        ax2.set_xlabel('Peak Cycle Number', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Peak Cycles\n(When capacity reaches maximum)',
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')

        plt.tight_layout()
        plt.savefig('capacity_rise_peak_distribution.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"图片已保存: capacity_rise_peak_distribution.png")
        plt.close()

        # ==================== 绘制单个典型案例的详细放大图 ====================
        # 选择回升幅度最大的电池
        best_battery = top_rise[0]
        battery_id = best_battery['id']
        result = results[battery_id]

        fig3, ax = plt.subplots(figsize=(10, 6))

        cycles = np.array(result['cycles'])
        soh = np.array(result['soh'])

        # 绘制完整曲线
        ax.plot(cycles, soh, 'lightgray', linewidth=1.5, alpha=0.6, label='Complete degradation curve')

        # 早期放大区域
        early_mask = cycles <= 500
        ax.plot(cycles[early_mask], soh[early_mask], 'b-', linewidth=2.5,
               label='Early cycles (0-500)')

        # 标注关键点
        peak_cycle = best_battery['peak_cycle']
        peak_soh = best_battery['peak_soh']
        initial_soh = best_battery['initial_soh']

        # 峰值点
        ax.scatter([peak_cycle], [peak_soh], color='red', s=150, zorder=5,
                  edgecolor='white', linewidth=2)
        ax.annotate(f'Peak\nCycle {peak_cycle}\nSOH {peak_soh:.2f}%',
                   xy=(peak_cycle, peak_soh), xytext=(peak_cycle+80, peak_soh+0.5),
                   fontsize=10, fontweight='bold', color='red',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

        # 初始点
        ax.scatter([1], [initial_soh], color='green', s=150, zorder=5,
                  edgecolor='white', linewidth=2)
        ax.annotate(f'Initial\nSOH {initial_soh:.2f}%',
                   xy=(1, initial_soh), xytext=(60, initial_soh-0.8),
                   fontsize=10, fontweight='bold', color='green',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        # 回升区域着色
        rise_mask = (cycles >= 1) & (cycles <= peak_cycle)
        ax.fill_between(cycles[rise_mask], initial_soh, soh[rise_mask],
                       alpha=0.3, color='lightgreen', label='Capacity rise region')

        # 添加回升幅度标注
        ax.annotate('', xy=(peak_cycle-30, peak_soh), xytext=(peak_cycle-30, initial_soh),
                   arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
        ax.text(peak_cycle-25, (peak_soh + initial_soh)/2,
               f'Rise: +{best_battery["rise_amplitude"]:.2f}%',
               fontsize=11, color='darkgreen', fontweight='bold', rotation=90, va='center')

        ax.set_xlim([0, 500])
        ax.set_ylim([min(soh[early_mask])*0.998, max(soh[early_mask])*1.003])
        ax.set_xlabel('Cycle Number', fontsize=12)
        ax.set_ylabel('SOH (%)', fontsize=12)
        ax.set_title(f'Battery {battery_id}: Early Capacity Rise Phenomenon (Zoomed)',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', fontsize=10)

        # 添加说明文本框
        textstr = (f'Battery: {battery_id}\n'
                  f'Initial SOH: {initial_soh:.2f}%\n'
                  f'Peak SOH: {peak_soh:.2f}%\n'
                  f'Peak Cycle: {peak_cycle}\n'
                  f'Rise Amplitude: +{best_battery["rise_amplitude"]:.2f}%')
        props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        plt.savefig('capacity_rise_detailed.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"图片已保存: capacity_rise_detailed.png")
        plt.close()

    return results, rise_batteries

if __name__ == '__main__':
    results, rise_batteries = main()
