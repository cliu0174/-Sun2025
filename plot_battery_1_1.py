"""
生成1-1电池的SOH、电流平均值、电压平均值随循环次数变化的三合一长图
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体和高清输出
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300

def load_battery_data(battery_id='1-1'):
    """加载电池数据"""
    with open(f'our_data/{battery_id}.pkl', 'rb') as f:
        data = pickle.load(f)
    return data[battery_id]

def compute_cycle_statistics(battery_data):
    """
    计算每个循环的统计量

    Returns:
        cycles: 循环号列表
        capacities: 容量列表
        current_means: 电流平均值列表
        voltage_means: 电压平均值列表
    """
    cycles = sorted(battery_data['dq'].keys())
    capacities = []
    current_means = []
    voltage_means = []

    for cycle_num in cycles:
        # 容量
        capacities.append(battery_data['dq'][cycle_num])

        # 获取该循环的数据
        if cycle_num in battery_data['data']:
            cycle_df = battery_data['data'][cycle_num]
            current_means.append(cycle_df['Current (mA)'].mean())
            voltage_means.append(cycle_df['Voltage (V)'].mean())
        else:
            current_means.append(np.nan)
            voltage_means.append(np.nan)

    return cycles, capacities, current_means, voltage_means

def plot_battery_triple(battery_id='1-1'):
    """
    绘制三合一长图：SOH、电流平均值、电压平均值 随循环次数变化

    Args:
        battery_id: 电池ID
    """
    # 加载数据
    battery_data = load_battery_data(battery_id)

    # 计算统计量
    cycles, capacities, current_means, voltage_means = compute_cycle_statistics(battery_data)

    # 转换单位
    cycles = np.array(cycles)
    capacities = np.array(capacities)
    current_means = np.array(current_means) / 1000  # mA -> A
    voltage_means = np.array(voltage_means)  # V

    # 计算SOH (相对于初始容量的百分比)
    initial_capacity = capacities[0]
    soh_values = capacities / initial_capacity * 100

    # 创建图形 - 三行一列的长图，增加间距
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    plt.subplots_adjust(hspace=0.35, left=0.10, right=0.95, top=0.94, bottom=0.06)

    # 主色调
    color_soh = '#2E86AB'      # 蓝色
    color_current = '#E94F37'  # 红色
    color_voltage = '#28A745'  # 绿色

    # ========== 子图1: SOH随循环次数变化 ==========
    ax1 = axes[0]
    ax1.plot(cycles, soh_values, color=color_soh, linewidth=1.5, alpha=0.9)
    ax1.fill_between(cycles, soh_values, alpha=0.15, color=color_soh)
    ax1.scatter(cycles[::50], soh_values[::50], color=color_soh, s=15, alpha=0.6, zorder=3)

    ax1.set_xlabel('Cycle Number', fontsize=11, labelpad=8)
    ax1.set_ylabel('SOH (%)', fontsize=11, labelpad=8)
    ax1.set_title(f'Battery {battery_id} - State of Health (SOH)',
                  fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([0, max(cycles)])
    ax1.set_ylim([min(soh_values)*0.98, 102])
    ax1.tick_params(axis='both', labelsize=10)

    # 添加容量信息文本框
    textstr = f'Initial: {initial_capacity:.1f} mAh\nFinal: {capacities[-1]:.1f} mAh\nRetention: {soh_values[-1]:.1f}%'
    props = dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax1.text(0.02, 0.97, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # ========== 子图2: 电流平均值随循环次数变化 ==========
    ax2 = axes[1]
    ax2.plot(cycles, current_means, color=color_current, linewidth=1.2, alpha=0.9)
    ax2.fill_between(cycles, current_means, alpha=0.12, color=color_current)
    ax2.scatter(cycles[::50], current_means[::50], color=color_current, s=15, alpha=0.6, zorder=3)

    ax2.set_xlabel('Cycle Number', fontsize=11, labelpad=8)
    ax2.set_ylabel('Average Current (A)', fontsize=11, labelpad=8)
    ax2.set_title(f'Battery {battery_id} - Average Current per Cycle',
                  fontsize=13, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([0, max(cycles)])
    ax2.tick_params(axis='both', labelsize=10)

    # 添加统计信息
    mean_current = np.nanmean(current_means)
    std_current = np.nanstd(current_means)
    textstr2 = f'Mean: {mean_current:.3f} A\nStd: {std_current:.3f} A'
    ax2.text(0.02, 0.97, textstr2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # ========== 子图3: 电压平均值随循环次数变化 ==========
    ax3 = axes[2]
    ax3.plot(cycles, voltage_means, color=color_voltage, linewidth=1.2, alpha=0.9)
    ax3.fill_between(cycles, voltage_means, alpha=0.12, color=color_voltage)
    ax3.scatter(cycles[::50], voltage_means[::50], color=color_voltage, s=15, alpha=0.6, zorder=3)

    ax3.set_xlabel('Cycle Number', fontsize=11, labelpad=8)
    ax3.set_ylabel('Average Voltage (V)', fontsize=11, labelpad=8)
    ax3.set_title(f'Battery {battery_id} - Average Voltage per Cycle',
                  fontsize=13, fontweight='bold', pad=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim([0, max(cycles)])
    ax3.tick_params(axis='both', labelsize=10)

    # 添加统计信息
    mean_voltage = np.nanmean(voltage_means)
    std_voltage = np.nanstd(voltage_means)
    textstr3 = f'Mean: {mean_voltage:.3f} V\nStd: {std_voltage:.3f} V'
    ax3.text(0.02, 0.97, textstr3, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # 保存图片
    output_path = f'battery_{battery_id.replace("-", "_")}_triple_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'图片已保存到: {output_path}')

    # 同时保存一个更高清的版本
    output_path_hd = f'battery_{battery_id.replace("-", "_")}_triple_plot_HD.png'
    plt.savefig(output_path_hd, dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f'高清版本已保存到: {output_path_hd}')

    plt.close(fig)

    return fig

if __name__ == '__main__':
    # 绘制1-1电池的三合一图
    fig = plot_battery_triple(battery_id='1-1')
