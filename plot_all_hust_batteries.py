"""
Plot all HUST battery capacity degradation curves overlaid in one figure.

Usage:
    # Plot all batteries
    python plot_all_hust_batteries.py
    
    # High resolution output
    python plot_all_hust_batteries.py --dpi 300
    
    # Custom save path
    python plot_all_hust_batteries.py --save-path my_results/all_batteries.png
    
    # High contrast mode (use more colors)
    python plot_all_hust_batteries.py --colormap tab20c
    
    # No display, save only
    python plot_all_hust_batteries.py --no-show
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

from src.utils import ensure_dir


def get_all_hust_batteries(data_dir='data/HUST data'):
    """
    Get list of all HUST batteries.
    
    Args:
        data_dir: Data directory path
        
    Returns:
        Sorted list of battery names
    """
    batteries = []
    
    if os.path.exists(data_dir):
        for file in sorted(os.listdir(data_dir)):
            if file.endswith('.csv'):
                battery_name = file.replace('.csv', '')
                batteries.append(battery_name)
    
    return sorted(batteries)


def plot_all_batteries_overlay(batteries, data_dir='data/HUST data',
                               figsize=(16, 10), colormap='tab20'):
    """
    Plot all batteries capacity degradation curves overlaid in one figure.
    
    Args:
        batteries: List of battery names
        data_dir: Data directory path
        figsize: Figure size
        colormap: Color map name
        
    Returns:
        (fig, ax, stats_dict)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors
    if colormap == 'tab20':
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    elif colormap == 'tab20c':
        colors = plt.cm.tab20c(np.linspace(0, 1, 20))
    else:
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(batteries)))
    
    stats_dict = {}
    loaded_count = 0
    
    print(f"\nLoading battery data...")
    print("-" * 70)
    
    for idx, battery_name in enumerate(batteries):
        data_path = os.path.join(data_dir, f'{battery_name}.csv')
        
        if not os.path.exists(data_path):
            print(f"[WARN] {battery_name} - File not found")
            continue
        
        try:
            # Read data
            df = pd.read_csv(data_path)
            capacity = df['capacity'].values
            cycles = np.arange(1, len(capacity) + 1)
            
            # Get color (cycle through colors)
            color = colors[idx % len(colors)]
            
            # Plot curve
            ax.plot(cycles, capacity, linewidth=1.5, label=battery_name,
                    color=color, alpha=0.8)
            
            # Save statistics
            stats_dict[battery_name] = {
                'cycles': cycles,
                'capacity': capacity,
                'initial': capacity[0],
                'final': capacity[-1],
                'loss': capacity[0] - capacity[-1],
                'loss_pct': ((capacity[0] - capacity[-1]) / capacity[0]) * 100
            }
            
            loaded_count += 1
            print(f"[OK] {battery_name:<8} | Init: {capacity[0]:.4f} Ah | "
                  f"Final: {capacity[-1]:.4f} Ah | "
                  f"Loss: {stats_dict[battery_name]['loss_pct']:.2f}% | "
                  f"Cycles: {len(capacity)}")
            
        except Exception as e:
            print(f"[ERROR] {battery_name} - Read error: {str(e)}")
            continue
    
    print("-" * 70)
    print(f"Successfully loaded: {loaded_count}/{len(batteries)} batteries\n")
    
    # Set labels and title
    ax.set_xlabel('Cycle Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Capacity (Ah)', fontsize=14, fontweight='bold')
    ax.set_title(f'All HUST Batteries Capacity Degradation ({loaded_count} batteries)',
                 fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend (split into 2-3 columns to avoid crowding)
    if loaded_count <= 20:
        ax.legend(fontsize=8, loc='best', ncol=2, framealpha=0.95)
    else:
        ax.legend(fontsize=7, loc='best', ncol=3, framealpha=0.95)
    
    # Add statistics info box
    avg_loss = np.mean([s['loss_pct'] for s in stats_dict.values()])
    max_cycles = max([len(s['cycles']) for s in stats_dict.values()])
    min_cycles = min([len(s['cycles']) for s in stats_dict.values()])
    
    stats_text = (
        f'Total Batteries: {loaded_count}\n'
        f'Avg Capacity Loss: {avg_loss:.2f}%\n'
        f'Max Cycles: {max_cycles}\n'
        f'Min Cycles: {min_cycles}'
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
            fontsize=11, family='monospace', fontweight='bold')
    
    plt.tight_layout()
    
    return fig, ax, stats_dict


def plot_all_batteries_with_statistics(batteries, data_dir='data/HUST data',
                                        figsize=(18, 12)):
    """
    Plot all batteries capacity curves with statistics (multiple subplots).
    
    Args:
        batteries: List of battery names
        data_dir: Data directory path
        figsize: Figure size
        
    Returns:
        (fig, stats_dict)
    """
    # Create figure with 4 subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # First subplot: all battery capacity curves
    ax1 = fig.add_subplot(gs[0, :])
    
    # Second subplot: initial capacity distribution
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Third subplot: capacity loss distribution
    ax3 = fig.add_subplot(gs[1, 1])
    
    stats_dict = {}
    initial_capacities = []
    capacity_losses = []
    battery_names_list = []
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    loaded_count = 0
    
    print(f"\nLoading all battery data...")
    print("-" * 70)
    
    for idx, battery_name in enumerate(batteries):
        data_path = os.path.join(data_dir, f'{battery_name}.csv')
        
        if not os.path.exists(data_path):
            continue
        
        try:
            df = pd.read_csv(data_path)
            capacity = df['capacity'].values
            cycles = np.arange(1, len(capacity) + 1)
            
            color = colors[idx % len(colors)]
            
            ax1.plot(cycles, capacity, linewidth=1.5, label=battery_name,
                    color=color, alpha=0.8)
            
            initial_cap = capacity[0]
            final_cap = capacity[-1]
            loss = initial_cap - final_cap
            loss_pct = (loss / initial_cap) * 100
            
            stats_dict[battery_name] = {
                'cycles': cycles,
                'capacity': capacity,
                'initial': initial_cap,
                'final': final_cap,
                'loss': loss,
                'loss_pct': loss_pct
            }
            
            initial_capacities.append(initial_cap)
            capacity_losses.append(loss_pct)
            battery_names_list.append(battery_name)
            
            loaded_count += 1
            print(f"[OK] {battery_name:<8} | Init: {initial_cap:.4f} Ah | "
                  f"Final: {final_cap:.4f} Ah | Loss: {loss_pct:.2f}%")
            
        except Exception as e:
            print(f"[ERROR] {battery_name}: {str(e)}")
            continue
    
    print("-" * 70)
    print(f"Successfully loaded: {loaded_count}/{len(batteries)} batteries\n")
    
    # Format first subplot
    ax1.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Capacity (Ah)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Capacity Degradation for {loaded_count} HUST Batteries',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=7, loc='best', ncol=3, framealpha=0.9)
    
    # Histogram: initial capacity distribution
    ax2.hist(initial_capacities, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Initial Capacity (Ah)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title(f'Initial Capacity Distribution (n={len(initial_capacities)})',
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics to histogram
    mean_init = np.mean(initial_capacities)
    std_init = np.std(initial_capacities)
    ax2.axvline(mean_init, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_init:.4f}')
    ax2.legend()
    
    # Histogram: capacity loss percentage distribution
    ax3.hist(capacity_losses, bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Capacity Loss (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title(f'Capacity Loss Distribution (n={len(capacity_losses)})',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics to histogram
    mean_loss = np.mean(capacity_losses)
    std_loss = np.std(capacity_losses)
    ax3.axvline(mean_loss, color='darkred', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.2f}%')
    ax3.legend()
    
    plt.tight_layout()
    
    return fig, stats_dict


def print_summary_statistics(stats_dict):
    """
    Print summary statistics for all batteries.
    
    Args:
        stats_dict: Dictionary with battery statistics
    """
    print("\n" + "=" * 80)
    print(" " * 25 + "HUST BATTERY STATISTICS SUMMARY")
    print("=" * 80)
    
    initial_capacities = [s['initial'] for s in stats_dict.values()]
    final_capacities = [s['final'] for s in stats_dict.values()]
    capacity_losses = [s['loss'] for s in stats_dict.values()]
    loss_percentages = [s['loss_pct'] for s in stats_dict.values()]
    cycle_counts = np.array([len(s['cycles']) for s in stats_dict.values()])
    
    print(f"\n{'Metric':<20} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std Dev':<15}")
    print("-" * 80)
    
    print(f"{'Initial Cap (Ah)':<20} {min(initial_capacities):<15.6f} {max(initial_capacities):<15.6f} "
          f"{np.mean(initial_capacities):<15.6f} {np.std(initial_capacities):<15.6f}")
    
    print(f"{'Final Cap (Ah)':<20} {min(final_capacities):<15.6f} {max(final_capacities):<15.6f} "
          f"{np.mean(final_capacities):<15.6f} {np.std(final_capacities):<15.6f}")
    
    print(f"{'Capacity Loss (Ah)':<20} {min(capacity_losses):<15.6f} {max(capacity_losses):<15.6f} "
          f"{np.mean(capacity_losses):<15.6f} {np.std(capacity_losses):<15.6f}")
    
    print(f"{'Loss Percent (%)':<20} {min(loss_percentages):<15.2f} {max(loss_percentages):<15.2f} "
          f"{np.mean(loss_percentages):<15.2f} {np.std(loss_percentages):<15.2f}")
    
    print(f"{'Cycle Count':<20} {int(np.min(cycle_counts)):<15} {int(np.max(cycle_counts)):<15} "
          f"{np.mean(cycle_counts):<15.1f} {np.std(cycle_counts):<15.1f}")
    
    print("-" * 80)
    print(f"Total batteries: {len(stats_dict)}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot all HUST batteries capacity degradation curves in one figure'
    )
    
    parser.add_argument('--data-dir', type=str, default='data/HUST data',
                        help='Data directory path')
    parser.add_argument('--save-dir', type=str, default='results/capacity_curves',
                        help='Directory to save plots')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Custom save path for overlay plot')
    parser.add_argument('--dpi', type=int, default=100,
                        help='DPI for saved figure')
    parser.add_argument('--colormap', type=str, default='tab20',
                        choices=['tab20', 'tab20c', 'tab10', 'hsv', 'jet'],
                        help='Colormap to use')
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 10],
                        help='Figure size: width height')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the figure')
    parser.add_argument('--include-stats', action='store_true',
                        help='Include detailed statistics subplots')
    parser.add_argument('--save-stats', type=str, default=None,
                        help='Save statistics to CSV file')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.save_dir)
    
    # Get all batteries
    print("\nScanning HUST data directory...")
    batteries = get_all_hust_batteries(args.data_dir)
    print(f"Found {len(batteries)} batteries")
    print(f"Battery list: {batteries}\n")
    
    print("=" * 72)
    
    if args.include_stats:
        print("Plot mode: Detailed mode (with statistics subplots)")
    else:
        print("Plot mode: Simple mode (all batteries overlaid)")
    
    print("=" * 72)
    
    # Generate overlay plot
    fig, ax, stats_dict = plot_all_batteries_overlay(
        batteries, args.data_dir,
        figsize=tuple(args.figsize),
        colormap=args.colormap
    )
    
    # Save overlay plot
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = os.path.join(args.save_dir, 'all_batteries_overlay.png')
    
    fig.savefig(save_path, dpi=args.dpi, bbox_inches='tight')
    print(f"\n[OK] Figure saved: {save_path}")
    
    # Generate detailed statistics plot if requested
    if args.include_stats:
        fig_stats, stats_dict_detail = plot_all_batteries_with_statistics(
            batteries, args.data_dir,
            figsize=(18, 12)
        )
        stats_save_path = os.path.join(args.save_dir, 'all_batteries_with_stats.png')
        fig_stats.savefig(stats_save_path, dpi=args.dpi, bbox_inches='tight')
        print(f"[OK] Detailed figure saved: {stats_save_path}")
    
    # Display if requested
    if not args.no_show:
        plt.show()
    
    # Print summary statistics
    print_summary_statistics(stats_dict)
    
    # Save statistics to CSV if requested
    if args.save_stats:
        try:
            df_stats = pd.DataFrame(stats_dict).T
            df_stats.to_csv(args.save_stats, float_format='%.6f')
            print(f"[OK] Statistics saved: {args.save_stats}")
        except Exception as e:
            print(f"[WARN] Error saving CSV: {str(e)}")
    
    print("[OK] Done!\n")


if __name__ == '__main__':
    main()
