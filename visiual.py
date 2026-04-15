import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ==========================================
# 1. 准备数据 (这里我手动提取了你图片中的部分关键数据作为示例)
# ==========================================
data = [
    # DropRate, Segments, ConstraintType, RMSE, IsBaseline
    (0.025, 10, 'Baseline', 1.2505, True),
    (0.025, 10, 'Best_Phy', 1.0629, False), # 对应 [0.1...]
    
    (0.025, 20, 'Baseline', 1.1442, True),
    (0.025, 20, 'Best_Phy', 1.0753, False),
    
    (0.05, 10, 'Baseline', 1.1339, True),
    (0.05, 10, 'Best_Phy', 1.1819, False), # 这里物理反而略差，如实记录
    
    (0.1, 10, 'Baseline', 1.3146, True),
    (0.1, 10, 'Best_Phy', 1.0639, False), # 显著提升
    
    (0.1, 20, 'Baseline', 1.0661, True),
    (0.1, 20, 'Best_Phy', 0.9311, False),
]

df = pd.DataFrame(data, columns=['Drop Rate', 'Segments', 'Type', 'RMSE', 'IsBaseline'])
df['Scenario'] = "DR:" + df['Drop Rate'].astype(str) + " / Seg:" + df['Segments'].astype(str)

# ==========================================
# 图表 1: 分组柱状图 (Baseline vs Best Physical)
# ==========================================
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# 创建柱状图
bar_plot = sns.barplot(
    data=df, 
    x='Scenario', 
    y='RMSE', 
    hue='Type', 
    palette={'Baseline': '#e74c3c', 'Best_Phy': '#2ecc71'},
    edgecolor='black'
)

# 添加提升百分比标签
for i in range(0, len(df), 2): # 假设数据是成对出现的
    baseline = df.iloc[i]['RMSE']
    best = df.iloc[i+1]['RMSE']
    improvement = (baseline - best) / baseline * 100
    
    # 在柱子上方标注
    if improvement > 0:
        plt.text(i//2 + 0.2, best + 0.02, f"↓{improvement:.1f}%", 
                 ha='center', color='green', fontweight='bold')
    else:
        plt.text(i//2 + 0.2, best + 0.02, f"↑{-improvement:.1f}%", 
                 ha='center', color='red')

plt.title('Performance Comparison: NoPhysicalLoss vs. Physical Constraints', fontsize=14)
plt.ylabel('RMSE (Lower is Better)')
plt.xlabel('Scenario (Drop Rate / Total Segments)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ==========================================
# 图表 2: 提升率热力图 (Heatmap)
# ==========================================
# 构造热力图数据结构：行是DropRate，列是Segments，值是RMSE提升率
# 这里手动填入你表格中黄色高亮的RMSE提升最大值
heatmap_data = np.array([
    [15.00, 6.02, 2.42],  # Drop Rate 0.025 (对应 Seg 10, 20, 50)
    [-4.23, 4.43, 2.44],  # Drop Rate 0.05 (注意有个负值)
    [19.07, 12.66, 11.85] # Drop Rate 0.1 (提升显著)
])

x_labels = [10, 20, 50]
y_labels = [0.025, 0.05, 0.1]

plt.figure(figsize=(8, 6))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".2f", 
    cmap="RdYlGn", # 红(差)-黄(中)-绿(好)
    center=0,      # 0设为中心，红色为负提升，绿色为正提升
    xticklabels=x_labels,
    yticklabels=y_labels,
    cbar_kws={'label': 'RMSE Improvement (%)'}
)

plt.title('Impact of Physical Constraints across Scenarios', fontsize=14)
plt.xlabel('Total Segments')
plt.ylabel('Drop Rate')
plt.tight_layout()
plt.show()