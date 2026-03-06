# 物理约束 - 快速开始

## 🚀 5分钟上手

### 1. 验证系统（30秒）

```bash
python test_physics_integration.py
```

**预期输出**：
```
✅ 测试 1: Windowing + Metadata - 通过
✅ 测试 2: Dataset 创建 - 通过
✅ 测试 3: DataLoader - 通过
✅ 测试 4: 物理损失计算 - 通过

所有测试通过！
```

### 2. 训练第一个模型（2分钟）

```bash
# 使用 10 个电池快速测试
python train_with_physics.py --model lstm --max_batteries 10
```

### 3. 对比实验（5分钟）

```bash
# 无物理约束（baseline）
python train_with_physics.py --model lstm --max_batteries 10 --no_physics

# 有物理约束
python train_with_physics.py --model lstm --max_batteries 10
```

---

## 📋 命令行参数

```bash
python train_with_physics.py \
  --model lstm           # 模型类型 (lstm/gru/bilstm/bigru)
  --device cuda          # 设备 (cuda/cpu)
  --max_batteries 10     # 电池数量限制
  --cleaning             # 启用数据清洗
  --no_physics           # 禁用物理约束（对比实验）
```

---

## ⚙️ 配置文件

编辑 `configs/models/lstm_config.json`：

```json
{
  "physics_constraints": {
    "enabled": true,          // 启用/禁用
    "monotonic_weight": 0.1,  // 单调性权重 (0.05-0.2)
    "monotonic_tolerance": 0.01,  // 容忍度 (0.005-0.02)
    "temporal_decay": {
      "max_step": 20,         // 最大步长 (10-30)
      "decay_alpha": 0.2      // 衰减系数 (0.1-0.3)
    }
  }
}
```

---

## 📊 查看结果

训练完成后，结果保存在：
```
results/physics_constraints/
├── lstm_with_physics.json
└── lstm_without_physics.json
```

---

## 🎯 推荐配置

### 快速测试
```bash
--max_batteries 10  # 10个电池，~2分钟
```

### 标准训练
```bash
--max_batteries 50  # 50个电池，~15分钟
```

### 完整训练
```bash
# 不加 --max_batteries，使用全部77个电池，~30分钟
```

---

## 📚 详细文档

- **使用指南**：[PHYSICS_CONSTRAINTS_USAGE.md](PHYSICS_CONSTRAINTS_USAGE.md)
- **实现总结**：[PHYSICS_CONSTRAINTS_SUMMARY.md](PHYSICS_CONSTRAINTS_SUMMARY.md)
- **实现方案**：[PHYSICS_CONSTRAINTS_IMPLEMENTATION_PLAN.md](PHYSICS_CONSTRAINTS_IMPLEMENTATION_PLAN.md)

---

**祝训练顺利！** 🎉
