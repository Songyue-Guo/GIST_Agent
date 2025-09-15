# 大模型间质瘤诊断准确率测评系统

## 概述

本系统用于评估各种大语言模型（LLM）在胃肠间质瘤（GIST）诊断任务上的准确性。系统支持多种主流大模型，包括OpenAI GPT、Claude、Azure GPT等，并提供详细的性能分析报告。

## 功能特性

### 🔬 支持的模型
- **OpenAI GPT**: GPT-4, GPT-3.5-turbo
- **Anthropic Claude**: Claude-3-Opus
- **Azure OpenAI**: Azure版GPT模型
- **模拟模型**: 用于测试和演示

### 📊 评估指标
- **总体准确率**: 所有预测正确的比例
- **精确率 (Precision)**: 针对间质瘤和非间质瘤分别计算
- **召回率 (Recall)**: 针对间质瘤和非间质瘤分别计算
- **F1分数**: 精确率和召回率的调和平均
- **混淆矩阵**: 包含TP、FP、TN、FN统计

### 💾 输出文件
- **详细预测结果**: `detailed_results_{model_name}_{timestamp}.csv`
- **评估摘要**: `evaluation_summary_{model_name}_{timestamp}.json`
- **模型比较报告**: `model_comparison_{timestamp}.csv`

## 安装要求

```bash
pip install pandas requests pathlib logging dataclasses
```

对于使用真实API的情况，还需要：
```bash
pip install openai  # 用于OpenAI模型
```

## 环境配置

### 设置API密钥

在运行脚本前，需要设置相应的环境变量：

```bash
# OpenAI API
export OPENAI_API_KEY="your_openai_api_key"

# Anthropic Claude API  
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Azure OpenAI (如果使用)
export AZURE_OPENAI_API_KEY="your_azure_api_key"
export AZURE_OPENAI_ENDPOINT="your_azure_endpoint"
```

## 使用方法

### 基本运行

```bash
cd hebei4/benchmark
python run_evaluation.py
```

### 数据准备

确保测试数据位于正确位置：
```
hebei4/
├── data/
│   └── train_data/
│       ├── 1.json
│       ├── 1_differential_diagnosis.json
│       └── ... (其他测试文件)
└── benchmark/
    └── run_evaluation.py
```

### 数据格式

测试数据应为JSON格式，包含以下字段：
```json
{
    "question": "患者的临床资料描述...",
    "think": "<think>推理过程...</think>", 
    "answer": "<answer>诊断结果</answer>"
}
```

## 诊断提示模板

系统使用专业的医学诊断提示模板：

```
你是一位经验丰富的消化内科专家，专门从事胃肠道肿瘤的诊断。请根据以下患者的临床资料，给出你的诊断结论。

重要说明：
1. 请仔细分析患者的症状、体征、影像学检查和内镜检查结果
2. 重点关注是否为胃肠间质瘤（GIST）
3. 请给出明确的诊断结论，格式为："诊断：[具体疾病名称]"
4. 如果是间质瘤，请明确指出；如果不是，请说明具体是什么疾病

患者临床资料：
{question}

请基于上述信息给出你的诊断：
```

## 标签提取规则

系统通过关键词匹配从诊断结果中提取标签：

- **间质瘤 (GIST)**: 包含"间质瘤"、"GIST"、"胃肠间质瘤"、"间叶源性肿瘤"
- **非间质瘤 (NON_GIST)**: 其他所有诊断

## 输出示例

### 控制台输出
```
============================================================
大模型间质瘤诊断准确率测评系统
============================================================
加载了 2 个测试案例
✓ 已配置OpenAI模型
准备评估 2 个模型

[1/2] 评估模型: OpenAI-gpt-4

==================================================
模型: OpenAI-gpt-4
总体准确率: 0.8500
间质瘤 - 精确率: 0.8000, 召回率: 0.9000, F1: 0.8421
非间质瘤 - 精确率: 0.9000, 召回率: 0.8000, F1: 0.8421
执行时间: 45.23秒
==================================================
```

### CSV详细结果
| file_name | question | true_label | predicted_label | prediction_text | correct |
|-----------|----------|------------|-----------------|-----------------|---------|
| 1.json | 患者女性，31岁... | GIST | GIST | 诊断：胃肠间质瘤 | True |

### JSON评估摘要
```json
{
  "模型名称": "OpenAI-gpt-4",
  "测试案例总数": 10,
  "正确预测数": 8,
  "总体准确率": "0.8000",
  "间质瘤_精确率": "0.8571",
  "间质瘤_召回率": "0.7500",
  "间质瘤_F1分数": "0.8000"
}
```

## 扩展功能

### 添加新模型

1. 创建新的模型类继承 `LLMInterface`
2. 实现 `generate_response()` 和 `get_model_name()` 方法
3. 在 `main()` 函数中添加模型实例

```python
class CustomModel(LLMInterface):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate_response(self, prompt: str) -> str:
        # 实现API调用逻辑
        pass
    
    def get_model_name(self) -> str:
        return "CustomModel-v1"
```

### 自定义评估指标

可以修改 `DiagnosisEvaluator` 类来添加新的评估指标或修改现有的计算逻辑。

## 注意事项

1. **API限制**: 注意各大模型提供商的API调用频率限制
2. **成本控制**: 真实API调用会产生费用，请合理控制测试规模
3. **网络稳定**: 确保网络连接稳定，API调用超时设置为60秒
4. **数据隐私**: 如使用真实患者数据，请确保符合数据保护规定

## 故障排除

### 常见问题

1. **ImportError**: 确保安装了所有必需的依赖包
2. **API调用失败**: 检查API密钥和网络连接
3. **数据加载失败**: 检查数据文件路径和格式
4. **结果保存失败**: 确保有写入权限

### 联系支持

如遇到问题，请检查日志输出获取详细错误信息。 