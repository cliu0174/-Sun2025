# 图1: 数据处理流程图 (Data Processing Pipeline)

## 请绘制一个流程图，展示从原始数据到模型输入的完整数据处理流程

### 组件和流程

**[起点] HUST Battery Dataset**
- 77个电池
- 每个电池包含多个充放电循环
- 每个循环有16维特征（电压、电流、温度等）

**↓**

**[步骤1] Cross-Battery Split**
- 训练集: 46个电池 (60%)
- 验证集: 15个电池 (20%)
- 测试集: 16个电池 (20%)
- **注释**: 电池不交叉，确保泛化能力

**↓**

**[步骤2] Sliding Window (对每个电池)**
- 窗口大小: 40 time steps
- 输入: 原始时间序列 [N_cycles × 16_features]
- 输出: 窗口化序列 [(N-40) × 40 × 16]
- **可视化**: 显示一个滑动窗口如何移动

**↓ 训练/验证 分支**

**[步骤3a] Triplet Sampling (仅训练/验证)**
- 从同一电池中提取连续的三个窗口
- 时间点: t, t+1, t+2 (step_k=1)
- 输出格式:
  ```
  x_1: [batch, 40, 16]  # 时间 t
  x_2: [batch, 40, 16]  # 时间 t+1
  x_3: [batch, 40, 16]  # 时间 t+2
  y_1, y_2, y_3: 对应的SOH真值
  cycle_indices: 用于mask计算
  ```
- **颜色**: 用三种颜色区分x_1, x_2, x_3

**↓ 测试 分支**

**[步骤3b] Single-Sample Mode (仅测试)**
- 每次使用单个窗口
- 输出格式:
  ```
  x: [batch, 40, 16]
  y: [batch, 1]
  ```
- **注释**: 避免数据泄露

**↓ (两个分支合并)**

**[终点] Ready for CNN-LSTM Model**
- 训练/验证: Triplet格式
- 测试: Single格式

---

## 视觉要求

### 布局
- 自上而下的流程图
- 使用矩形框表示数据状态
- 使用菱形框表示决策点（训练/验证 vs 测试）
- 箭头标注数据shape变化

### 颜色编码
- 原始数据: 浅蓝色
- 训练/验证路径: 绿色
- 测试路径: 橙色
- Triplet三元组: 分别用深绿、中绿、浅绿

### 关键标注
1. 在"Cross-Battery Split"处标注: ⭐ 无电池交叉
2. 在"Triplet Sampling"处标注: ⭐ 仅用于训练/验证
3. 在每个数据转换处显示shape: [batch, seq, features]
4. 在"Sliding Window"处画一个小示意图展示窗口滑动

### 示例文字
- 在Triplet Sampling框内：
  ```
  连续三个时间点 (t, t+1, t+2)
  来自同一电池，确保物理连续性
  ```

---

## 输出格式建议
- Mermaid flowchart
- 或专业的架构图工具（Lucidchart, draw.io风格）
- 清晰、专业、适合论文使用
