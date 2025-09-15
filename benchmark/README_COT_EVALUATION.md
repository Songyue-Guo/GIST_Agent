# 基于CoT上下文的大模型诊断评估系统

## 📖 概述

本系统提供了三个评估脚本，用于测试和对比大模型在胃肠道肿瘤诊断任务上的表现：

1. **`run_evaluation.py`** - 原版评估脚本（无CoT上下文）
2. **`run_evaluation_with_cot.py`** - CoT增强评估脚本（使用CoT示例作为上下文）
3. **`compare_evaluation.py`** - 对比评估脚本（同时运行两种方法并对比结果）

## 🎯 支持的病种

- **胃肠间质瘤 (GIST)**
- **平滑肌瘤 (LEIOMYOMA)**
- **异位胰腺 (PANCREATIC)**
- **神经鞘瘤 (SCHWANNOMA)**

## 📁 文件结构

```
hebei4/benchmark/
├── run_evaluation.py              # 原版评估脚本
├── run_evaluation_with_cot.py     # CoT增强评估脚本
├── compare_evaluation.py          # 对比评估脚本
├── README_COT_EVALUATION.md       # 本说明文档
└── results/                       # 评估结果输出目录
    ├── detailed_results_*.csv     # 详细预测结果
    ├── *_evaluation_summary_*.json # 评估摘要
    └── evaluation_comparison_*.json # 对比评估结果
```

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的Python包：
```bash
pip install openai pandas pathlib logging dataclasses
```

### 2. 配置API

脚本默认使用以下Azure OpenAI配置：
- **API Key**: `dbfa63b54a2744d7aba5c2008b125a86`
- **Endpoint**: `https://mdi-gpt-4o.openai.azure.com/`
- **Model**: `gpt-4o-global`

如需修改，请编辑各脚本中的`AzureModel`类初始化参数。

### 3. 运行评估

#### 方式一：原版评估（无CoT上下文）
```bash
cd hebei4/benchmark
python run_evaluation.py
```

#### 方式二：CoT增强评估
```bash
cd hebei4/benchmark
python run_evaluation_with_cot.py
```

#### 方式三：对比评估（推荐）⭐
```bash
cd hebei4/benchmark
python compare_evaluation.py
```

## 📊 CoT增强评估原理

### Few-Shot Learning with CoT

CoT增强评估采用few-shot learning的方式，为每个测试案例提供相同病种的高质量诊断示例：

```
你是一位经验丰富的消化内科专家...

**参考诊断示例：**

**示例 1：**
患者情况：患者为66岁女性，主诉体检发现胃肿物2月余...
诊断推理：1. 临床背景与特征初筛...
诊断结论：胃神经鞘瘤

**示例 2：**
患者情况：患者，女性，59岁，已婚，主诉为恶心、呕吐9天...
诊断推理：1. 临床与解剖学特征初筛...
诊断结论：平滑肌瘤

---

**当前需要诊断的患者情况：**
[实际测试案例]
```

### CoT示例来源

CoT示例来自于`../data/train_data/cot_data_2025_7_7/jsonl/`目录下的最新数据文件，包含：
- 高质量的医学推理过程
- 标准化的诊断结论
- 分病种分类的示例库

## 📈 评估指标

系统计算以下指标：

### 整体指标
- **准确率 (Accuracy)**: 正确预测数 / 总测试案例数
- **执行时间**: 完成所有测试案例的时间
- **CoT示例使用数**: 总共使用的CoT示例数量

### 各病种指标
- **精确率 (Precision)**: TP / (TP + FP)
- **召回率 (Recall)**: TP / (TP + FN)
- **F1分数**: 2 × Precision × Recall / (Precision + Recall)

## 📋 输出文件说明

### 1. 详细结果文件 (`*_detailed_results_*.csv`)
包含每个测试案例的详细预测结果：
- `file_name`: 测试案例文件名
- `question`: 患者情况描述（截断版）
- `true_label`: 真实标签
- `predicted_label`: 模型预测标签
- `prediction_text`: 模型原始响应
- `correct`: 是否预测正确
- `cot_examples_used`: 使用的CoT示例数（仅CoT评估）

### 2. JSONL格式结果 (`*_detailed_results_*.jsonl`)
每行一个JSON对象，包含完整的测试案例信息，便于后续分析。

### 3. 评估摘要 (`*_evaluation_summary_*.json`)
包含整体和各病种的详细指标统计。

### 4. 对比评估结果 (`evaluation_comparison_*.json`)
包含原版和CoT增强评估的详细对比分析。

## ⚙️ 参数调整

### CoT示例数量
在`run_evaluation_with_cot.py`或`compare_evaluation.py`中调整：
```python
# 每个测试案例使用的CoT示例数量
num_cot_examples = 2  # 可以调整为1-5
```

### 测试案例数量限制
用于快速测试：
```python
# 限制测试案例数量（None表示使用全部）
max_test_cases = 10  # 设置为小数字进行快速测试
```

### API调用间隔
避免API限制：
```python
time.sleep(0.5)  # 可以调整间隔时间
```

## 🔍 使用示例

### 快速测试（推荐首次使用）
```python
# 在 compare_evaluation.py 的 main() 函数中修改：
comparison_result = comparator.run_comparison(
    num_cot_examples=2,      # 每案例使用2个CoT示例
    max_test_cases=5         # 仅测试5个案例
)
```

### 完整评估
```python
comparison_result = comparator.run_comparison(
    num_cot_examples=3,      # 每案例使用3个CoT示例
    max_test_cases=None      # 使用所有测试案例
)
```

## 📊 期望结果

根据经验，CoT增强评估通常能够带来：
- **准确率提升**: 2-10%
- **F1分数改善**: 各病种均有不同程度提升
- **推理质量**: 更符合临床思维模式
- **执行时间**: 略有增加（因为prompt更长）

## 🛠️ 故障排除

### 1. CoT文件不存在
```
错误: CoT文件不存在
解决: 确保 ../data/train_data/cot_data_2025_7_7/jsonl/ 目录存在相关文件
```

### 2. 测试数据加载失败
```
错误: 没有找到测试数据
解决: 检查 ../data/train_data/cot_data_gist/ 和 ../data/train_data/cot_data_n_gist/ 目录
```

### 3. API调用失败
```
错误: API调用失败
解决: 检查网络连接和API密钥配置
```

### 4. 内存不足
```
错误: 内存不足
解决: 降低 max_test_cases 参数或减少 num_cot_examples
```

## 📝 注意事项

1. **首次运行建议使用小数据集**测试，确保系统正常工作
2. **API调用会产生费用**，请合理控制测试规模
3. **结果文件会自动保存**，请定期清理results目录
4. **CoT示例质量直接影响评估效果**，确保使用高质量的CoT数据

## 🎯 最佳实践

1. **先运行对比评估**了解CoT效果
2. **调整CoT示例数量**找到最优配置
3. **分析详细结果文件**了解具体改进点
4. **定期更新CoT示例库**保持数据质量

---

## 📞 技术支持

如遇问题，请检查：
1. 文件路径是否正确
2. API配置是否有效
3. Python环境是否完整
4. 数据文件是否存在

**Happy Evaluating! 🚀** 