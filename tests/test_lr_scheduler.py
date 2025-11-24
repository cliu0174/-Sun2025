"""
测试学习率调度器是否正确工作
"""
import numpy as np
import matplotlib.pyplot as plt

# 模拟学习率调度
num_epochs = 200
warmup_epochs = 30
warmup_lr = 0.002
base_lr = 0.01
final_lr = 0.0002

learning_rates = []

for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        # Warmup阶段: 线性增长
        lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        # Cosine Decay阶段
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        lr = final_lr + (base_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    learning_rates.append(lr)

    # 打印关键epoch的学习率
    if epoch in [0, 10, 20, 29, 30, 50, 100, 150, 199]:
        print(f"Epoch {epoch+1:3d}: LR = {lr:.6f}")

# 绘制学习率曲线
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_epochs+1), learning_rates, linewidth=2)
plt.axvline(x=warmup_epochs, color='r', linestyle='--', label=f'Warmup ends (epoch {warmup_epochs})')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Warmup + Cosine Decay Learning Rate Schedule', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 添加关键点标注
plt.scatter([1, warmup_epochs, num_epochs],
           [learning_rates[0], learning_rates[warmup_epochs-1], learning_rates[-1]],
           color='red', s=100, zorder=5)
plt.text(1, learning_rates[0]+0.0005, f'Start: {learning_rates[0]:.4f}', ha='left')
plt.text(warmup_epochs, learning_rates[warmup_epochs-1]+0.0005, f'Peak: {learning_rates[warmup_epochs-1]:.4f}', ha='center')
plt.text(num_epochs, learning_rates[-1]+0.0005, f'Final: {learning_rates[-1]:.4f}', ha='right')

plt.tight_layout()
plt.savefig('lr_schedule_visualization.png', dpi=150)
print(f"\n学习率曲线已保存到: lr_schedule_visualization.png")
print(f"\n总结:")
print(f"  起始学习率 (epoch 1):     {learning_rates[0]:.6f}")
print(f"  峰值学习率 (epoch 30):    {learning_rates[warmup_epochs-1]:.6f}")
print(f"  最终学习率 (epoch 200):   {learning_rates[-1]:.6f}")
print(f"  Warmup阶段: Epoch 1-{warmup_epochs}")
print(f"  Cosine Decay阶段: Epoch {warmup_epochs+1}-{num_epochs}")
