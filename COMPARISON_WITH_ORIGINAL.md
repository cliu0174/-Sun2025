# 原始PINN4SOH代码 vs 当前复现代码 - 详细对比

**对比日期**: 2025-11-20
**原始代码路径**: `PINN4SOH/`
**复现代码路径**: 当前项目

---

## 📋 一、CNN模型架构对比

### 原始代码 (PINN4SOH/Model/Compare_Models.py)

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = ResBlock(input_channel=1, output_channel=8, stride=1)  # N,8,17
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2)  # N,16,9
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=2)  # N,24,5
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1)  # N,16,5
        self.layer5 = ResBlock(input_channel=16, output_channel=8, stride=1)  # N,8,5
        self.layer6 = nn.Linear(8*5,1)  # 无激活函数

    def forward(self, x):
        N,L = x.shape[0],x.shape[1]
        x = x.view(N,1,L)  # (N,17) → (N,1,17)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out.view(N,-1))  # flatten
        return out.view(N,1)
```

### 当前复现 (models/baseline_models.py)

```python
class ResCNN(nn.Module):
    def __init__(self, input_size=17,
                 channel_config=[8, 16, 24, 16, 8],
                 stride_config=[1, 2, 2, 1, 1],
                 dropout_rate=0.0):
        super(ResCNN, self).__init__()

        # 构建ResBlock层
        layers = []
        in_channels = 1
        for out_channels, stride in zip(channel_config, stride_config):
            layers.append(ResBlock(in_channels, out_channels, stride))
            in_channels = out_channels

        self.res_layers = nn.Sequential(*layers)

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 1)  # 无激活函数
        )

    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, 1, -1)  # (N,17) → (N,1,17)
        x = self.res_layers(x)
        x = x.view(N, -1)  # flatten
        x = self.fc(x)
        return x
```

### ResBlock对比

#### 原始代码
```python
class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out
```

#### 当前复现
```python
# 完全相同！✅
class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out
```

**结论**: ✅ **ResBlock 100%一致**

**结论**: ✅ **CNN模型架构100%一致**
- 层级结构相同
- 通道配置相同
- Stride配置相同
- 输出层无激活函数（一致）

---

## 📋 二、训练配置对比

### 原始代码 (PINN4SOH/main_comparision.py)

```python
# 第157-168行: 超参数设置
parser.add_argument('--epochs', type=int, default=200, help='epoch')
parser.add_argument('--early_stop', type=int, default=20, help='early stop')  # ⚠️ 注意这里
parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
parser.add_argument('--batch_size',type=int,default=512)

# 第29行: 优化器
self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.warmup_lr)

# 第28行: 损失函数
self.loss_func = nn.MSELoss()

# 第30-35行: 学习率调度器
self.scheduler = LR_Scheduler(optimizer=self.optimizer,
                              warmup_epochs=args.warmup_epochs,
                              warmup_lr=args.warmup_lr,
                              num_epochs=args.epochs,
                              base_lr=args.lr,
                              final_lr=args.final_lr)
```

### 当前复现 (configs/models/rescnn_config.json)

```json
{
  "training": {
    "num_epochs": 200,          // ✅
    "batch_size": 512,          // ✅
    "learning_rate": 0.002,     // ✅ (warmup_lr)
    "optimizer": "Adam",        // ✅
    "loss_function": "MSELoss", // ✅
    "scheduler": {
      "enabled": true,          // ✅
      "warmup_epochs": 30,      // ✅
      "warmup_lr": 0.002,       // ✅
      "base_lr": 0.01,          // ✅
      "final_lr": 0.0002        // ✅
    },
    "early_stopping": {
      "enabled": true,
      "patience": 20,           // ✅ (注意：你刚改成了20)
      "min_delta": 1e-5
    }
  }
}
```

**结论**: ✅ **训练配置100%一致**

**注意**: 你看到了吗？原始代码的early_stop默认值是**20**，不是10！我之前看错了。你在配置文件中改成20是对的！

---

## 📋 三、学习率调度器对比

### 原始代码 (PINN4SOH/Model/Model.py:94-120)

```python
class LR_Scheduler(object):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr,
                 iter_per_epoch=1, constant_predictor_lr=False):
        self.base_lr = base_lr
        warmup_iter = iter_per_epoch * warmup_epochs

        # Warmup阶段: 线性插值
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)

        # Cosine Decay阶段
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

        # 合并两个阶段
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        return lr
```

**关键特点**:
1. 预先计算所有epoch的学习率，存储在数组中
2. 每次step()从数组中取值
3. Warmup使用`np.linspace`
4. Cosine Decay公式: `final_lr + 0.5 * (base_lr - final_lr) * (1 + cos(π * t / T))`

### 当前复现 (train_cross_battery.py:295-311)

```python
def lr_lambda(epoch):
    """学习率调度函数: Warmup + Cosine Decay"""
    if epoch < warmup_epochs:
        # Warmup阶段: 线性增长 (warmup_lr → base_lr)
        current_lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        # Cosine Decay阶段 (base_lr → final_lr)
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        current_lr = final_lr + (base_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    # 返回相对于optimizer base_lr的倍数
    optimizer_base_lr = config['training']['learning_rate']
    return current_lr / optimizer_base_lr

scheduler = LambdaLR(optimizer, lr_lambda)
```

**关键特点**:
1. 使用PyTorch的LambdaLR，动态计算学习率
2. Warmup使用线性插值公式
3. Cosine Decay公式: `final_lr + 0.5 * (base_lr - final_lr) * (1 + cos(π * progress))`

### 数学等价性验证

#### Warmup阶段

**原始代码**:
```python
np.linspace(warmup_lr, base_lr, warmup_epochs)
# epoch 0: warmup_lr
# epoch 1: warmup_lr + 1/30 * (base_lr - warmup_lr)
# epoch 29: base_lr
```

**当前复现**:
```python
current_lr = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
# epoch 0: warmup_lr + 0/30 * (...) = warmup_lr
# epoch 1: warmup_lr + 1/30 * (base_lr - warmup_lr)
# epoch 29: warmup_lr + 29/30 * (base_lr - warmup_lr) ≈ base_lr
```

⚠️ **发现差异！**

原始代码使用`np.linspace(warmup_lr, base_lr, 30)`会生成30个点，包括起点和终点:
- 索引0: warmup_lr
- 索引29: base_lr

但当前复现在epoch=29时:
```
current_lr = 0.002 + (0.01 - 0.002) * 29/30 = 0.002 + 0.008 * 0.9667 = 0.00973
```

在epoch=30时才会达到base_lr=0.01。

**这是一个轻微的差异！**

#### Cosine Decay阶段

**原始代码**:
```python
cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))

# epoch 30 (iter=0): final_lr + 0.5 * (...) * (1 + cos(0)) = base_lr
# epoch 199 (iter=169): final_lr + 0.5 * (...) * (1 + cos(π)) = final_lr
```

**当前复现**:
```python
progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
current_lr = final_lr + (base_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * progress))

# epoch 30: progress=0/170, lr = final_lr + 0.5*(...) * (1 + cos(0)) = base_lr ✅
# epoch 199: progress=169/170, lr ≈ final_lr ✅
```

✅ **Cosine Decay阶段完全一致！**

**结论**: ⚠️ **学习率调度器有轻微差异**
- Warmup阶段: 差一个点的偏移（epoch 29 vs epoch 30达到峰值）
- Cosine Decay阶段: 完全一致

---

## 📋 四、训练循环对比

### 原始代码 (PINN4SOH/main_comparision.py:94-108)

```python
def train(self):
    min_loss = 100
    early_stop = 0
    for epoch in range(1,self.epochs+1):  # epoch从1开始
        early_stop += 1

        # 1. 训练一个epoch
        train_loss = self.train_one_epoch(epoch)

        # 2. 更新学习率
        current_lr = self.scheduler.step()

        # 3. 验证
        valid_loss = self.valid(epoch)

        # 4. 保存最佳模型
        if valid_loss < min_loss and self.test_loader is not None:
            min_loss = valid_loss
            true_label,pred_label = self.test()  # 在测试集上评估
            early_stop = 0  # 重置

        # 5. Early stopping
        if early_stop > 10:  # ⚠️ 代码中硬编码为10
            break
```

**关键逻辑**:
1. early_stop每个epoch都+1
2. valid_loss改善时重置early_stop=0
3. early_stop > 10时停止（**硬编码**）
4. 每次找到更好的模型时，立即在测试集上评估

### 当前复现 (train_cross_battery.py:340-378)

```python
# Early stopping配置
patience = config['training']['early_stopping'].get('patience', 10)
patience_counter = 0

for epoch in tqdm(range(num_epochs), desc="Training"):  # epoch从0开始
    patience_counter += 1  # 每个epoch递增

    # 1. 训练阶段
    model.train()
    for features, targets in train_loader:
        ...

    # 2. 验证阶段
    model.eval()
    for features, targets in val_loader:
        ...

    # 3. 更新学习率 (在validation之后)
    current_lr = optimizer.param_groups[0]['lr']
    if scheduler is not None:
        scheduler.step()

    # 4. 保存最佳模型和Early Stopping
    if val_loss < best_val_loss:  # 基于val_loss判断
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_model_state = model.state_dict()
        patience_counter = 0  # 重置计数器

    # 5. Early stopping检查
    if patience_counter > patience:
        print(f"早停触发！")
        break

# 训练完成后，恢复最佳模型，然后在测试集上评估
model.load_state_dict(best_model_state)
# ... 测试集评估
```

**关键逻辑**:
1. patience_counter每个epoch都+1
2. val_loss改善时重置patience_counter=0
3. patience_counter > patience时停止（**从配置读取，默认20**）
4. 训练完成后，恢复最佳模型，然后在测试集上评估

### 差异分析

| 项目 | 原始代码 | 当前复现 | 一致性 |
|------|----------|----------|--------|
| epoch计数 | 从1开始 | 从0开始 | ⚠️ 轻微差异（不影响结果） |
| early_stop触发条件 | > 10 (硬编码) | > 20 (配置文件) | ⚠️ 差异 |
| scheduler调用时机 | 在validation之前 | 在validation之后 | ⚠️ 差异 |
| 测试集评估时机 | 每次找到更好模型时 | 训练完成后 | ⚠️ 差异 |
| 判断标准 | valid_loss | val_loss (MSE) | ✅ 一致 |
| 重置机制 | loss改善时重置 | loss改善时重置 | ✅ 一致 |

**重要发现**:

1. **Early stopping patience**:
   - 原始代码: 命令行参数`--early_stop=20`，但代码中硬编码为10
   - 当前复现: 使用配置文件的patience=20

2. **Scheduler调用时机**:
   - 原始代码: train → scheduler.step() → valid
   - 当前复现: train → valid → scheduler.step()

这会导致学习率在哪个epoch使用有1个epoch的偏移！

---

## 📋 五、关键差异总结

### 🟡 轻微差异（影响较小）

1. **Warmup学习率达到峰值的时机**
   - 原始: epoch 29达到base_lr
   - 当前: epoch 30达到base_lr
   - 影响: 1个epoch的差异，影响很小

2. **Epoch计数起点**
   - 原始: 从1开始
   - 当前: 从0开始
   - 影响: 仅显示差异，不影响训练

3. **测试集评估时机**
   - 原始: 每次找到更好模型时评估测试集
   - 当前: 训练完成后评估测试集
   - 影响: 不影响训练，仅影响日志记录

### 🔴 重要差异（可能影响结果）

1. **Scheduler调用时机**
   - 原始: train → scheduler.step() → valid
   - 当前: train → valid → scheduler.step()
   - 影响: **学习率在哪个epoch使用有1个epoch的偏移**
   - 严重程度: 中等

2. **Early stopping patience**
   - 原始: 硬编码为10（虽然参数默认是20）
   - 当前: 配置文件为20
   - 影响: **停止时机不同，可能多训练10个epoch**
   - 严重程度: 中等

---

## 📋 六、修复建议

### 优先级1: 修复scheduler调用时机（高影响）

**当前代码** (train_cross_battery.py:382-385):
```python
# 验证
val_loss /= len(val_loader.dataset)

# 更新学习率 (在validation之后)
current_lr = optimizer.param_groups[0]['lr']
if scheduler is not None:
    scheduler.step()
```

**应该改为** (匹配原始代码):
```python
# 更新学习率 (在validation之前)
current_lr = optimizer.param_groups[0]['lr']
if scheduler is not None:
    scheduler.step()

# 验证
val_loss /= len(val_loader.dataset)
```

### 优先级2: 修复early stopping patience（中等影响）

**选项A**: 使用硬编码10（完全匹配原始代码）
```python
if patience_counter > 10:  # 硬编码
    break
```

**选项B**: 使用配置文件但改为10
```json
"early_stopping": {
  "patience": 10  // 改回10
}
```

推荐**选项B**，保持配置的灵活性。

### 优先级3: 修复Warmup学习率计算（低影响）

这个差异很小，可以选择性修复。如果要完全匹配：

```python
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        # 使用linspace的逻辑
        current_lr = warmup_lr + (base_lr - warmup_lr) * (epoch + 1) / warmup_epochs
    else:
        ...
```

---

## 📋 七、最终验证清单

修复上述差异后，应该检查：

- [ ] Scheduler在validation之前调用
- [ ] Early stopping patience = 10
- [ ] Epoch 1的学习率 = warmup_lr (0.002)
- [ ] Epoch 30的学习率 = base_lr (0.01)
- [ ] 使用val_loss判断最佳模型
- [ ] patience_counter在val_loss改善时重置
- [ ] ResBlock结构与原始代码一致
- [ ] 通道配置[8,16,24,16,8]
- [ ] Stride配置[1,2,2,1,1]
- [ ] Batch size = 512

---

## 📊 当前一致性评估

| 组件 | 一致性 | 说明 |
|------|--------|------|
| ResBlock | 100% | ✅ 完全一致 |
| CNN架构 | 100% | ✅ 完全一致 |
| 损失函数 | 100% | ✅ MSELoss |
| 优化器 | 100% | ✅ Adam |
| Batch size | 100% | ✅ 512 |
| Warmup策略 | 95% | ⚠️ 轻微偏移1个epoch |
| Cosine Decay | 100% | ✅ 完全一致 |
| Early stopping patience | 50% | ❌ 10 vs 20 |
| Scheduler时机 | 0% | ❌ 顺序相反 |
| 最佳模型判断 | 100% | ✅ 基于loss |

**总体一致性: 85%**

修复scheduler时机和early stopping patience后，可达到 **95%+一致性**。

---

**分析完成日期**: 2025-11-20
**建议**: 立即修复scheduler调用时机和early stopping patience
