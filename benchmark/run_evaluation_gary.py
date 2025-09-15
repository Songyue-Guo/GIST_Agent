#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于CoT上下文的大模型间质瘤等病种诊断准确率测评脚本
通过读取CoT jsonl文件作为few-shot learning上下文来提升模型表现
支持间质瘤、平滑肌瘤、异位胰腺、神经鞘瘤等多分类评估
"""

import json
import os
import re
import time
import pandas as pd
import random  # 新增
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import argparse
import openai
from jinja2 import Template
from transformers import AutoTokenizer
from openai import AzureOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalModel:
    """本地部署模型类（通过OpenAI兼容API调用）"""
    def __init__(self, model_name: str, port: int = 8168, use_chat_template: bool = True):
        self.model_name = model_name
        self.port = port
        self.use_chat_template = use_chat_template
        
        # 初始化API客户端
        self.client = openai.Client(
            base_url=f"http://10.120.20.177:{port}/v1",
            api_key="token-abc123",
        )
        
        # 初始化tokenizer和template（如果使用chat_template）
        if self.use_chat_template:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
            self.template = Template(self.tokenizer.chat_template)
        
    def postprocess_output(self, pred):
        """后处理模型输出"""
        pred = pred.replace("</s>", "")
        if len(pred) > 0 and pred[0] == " ":
            pred = pred[1:]
        return pred
        
    def generate_response(self, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.1) -> str:
        """生成模型响应"""
        try:
            # 如果使用chat_template，格式化输入
            if self.use_chat_template:
                formatted_prompt = self.template.render(
                    messages=[{"role": "user", "content": prompt}],
                    bos_token=self.tokenizer.bos_token,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt
            
            response = self.client.completions.create(
                model=self.model_name,
                prompt=[formatted_prompt],
                temperature=temperature, 
                top_p=0.9, 
                max_tokens=max_new_tokens
            )
            
            result = response.choices[0].text
            logger.info("===" * 64)
            logger.info(result)
            return self.postprocess_output(result)
            
        except Exception as e:
            logger.error(f"模型调用失败: {e}")
            return f"错误: {e}"


# class AzureModel:
#     """Azure OpenAI模型类"""
#     def __init__(self, api_key: str = 'dbfa63b54a2744d7aba5c2008b125a86', 
#                  endpoint: str ='https://mdi-gpt-4o.openai.azure.com/', 
#                  model_name: str = "gpt-4o-global"):
#         self.api_key = api_key
#         self.endpoint = endpoint
#         self.model_name = "gpt-4o-global"
        
#     def generate_response(self, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.1) -> str:
#         client = AzureOpenAI(
#             api_key=self.api_key,
#             api_version="2024-02-01",
#             azure_endpoint=self.endpoint,
#         )
#         response = client.chat.completions.create(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature,
#             max_tokens=max_new_tokens
#         )   
#         return response.choices[0].message.content or ""
    
@dataclass
class EvaluationResult:
    """评估结果数据类"""
    model_name: str
    total_cases: int
    correct_predictions: int
    accuracy: float
    per_class_metrics: Dict[str, Dict[str, float]]  # 每个类别的precision, recall, f1
    execution_time: float
    cot_examples_used: int  # 新增CoT示例使用统计

class DiagnosisEvaluator:
    """基于CoT上下文的多病种诊断准确率评估器"""
    
    def __init__(self, data_dir: str = "../data/train_data", 
                 cot_file_path: str = "../data/train_data/cot_data_2025_7_7/jsonl/cot_data_2025_7_7_20250723_155135_all_diseases_final.jsonl"):
        self.data_dir = Path(data_dir)
        self.num_cot_examples = 
        self.prompt_template = self._create_prompt_template(num_cot_examples=0, sampling_method='proportional')
        self.cot_file_path = cot_file_path
        self.cot_examples = {}  # 按病种分类的CoT示例
        
        # 定义病种类别
        self.disease_categories = {
            'GIST': ['间质瘤', 'GIST', '胃肠间质瘤'],
            'LEIOMYOMA': ['平滑肌瘤', '胃平滑肌瘤'],
            'PANCREATIC': ['异位胰腺', '胰腺异位', '迷走胰腺'],
            'SCHWANNOMA': ['神经鞘瘤', '施万细胞瘤'],
            'OTHER': []  # 其他疾病
        }
        
        # 病种映射（与原数据生成器一致）
        self.disease_mapping = {
            "胃间质瘤": "GIST",
            "神经鞘瘤": "SCHWANNOMA", 
            "胃平滑肌瘤": "LEIOMYOMA",
            "异位胰腺": "PANCREATIC"
        }
        
        # 加载CoT示例
        self.load_cot_examples()
        
    def _create_prompt_template(self, num_examples: int = 0, sampling_method: str = 'proportional') -> str:
        """创建包含CoT示例的诊断提示模板"""
        
        # 基础模板
        base_template = """你是一位经验丰富的消化内科专家，专门从事胃肠道肿瘤的诊断。请根据以下患者的临床资料，给出你的诊断结论。

请按照以下格式输出：
## Thinking
[请在这里进行详细的临床分析，包括症状分析、影像学特征分析、鉴别诊断等]

## Final Response
[请在这里给出明确的诊断结论，仅包含疾病名称，如：异位胰腺、神经鞘瘤、胃肠间质瘤、平滑肌瘤等]

"""
        
        # 如果需要添加CoT示例
        if num_examples > 0:
            cot_examples = self.get_random_cot_examples(num_examples, sampling_method)
            
            if cot_examples:
                base_template += "**参考诊断示例：**\n\n"
                
                for i, example in enumerate(cot_examples, 1):
                    # 清理think内容，移除HTML标签
                    think_content = re.sub(r'<think>|</think>', '', example['think']).strip()
                    answer_content = re.sub(r'<answer>|</answer>', '', example['answer']).strip()
                    
                    base_template += f"**示例 {i}：**\n"
                    base_template += f"患者情况：{example['question'][:150]}...\n"
                    base_template += f"诊断推理：{think_content[:200]}...\n"
                    base_template += f"诊断结论：{answer_content}\n\n"
                
                base_template += "---\n\n"
        
        base_template += "患者临床资料：\n{question}\n"
        
        return base_template
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        test_data = []
        
        # 加载间质瘤数据（从JSONL文件）
        gist_jsonl_files = list(self.data_dir.glob("*gist*final.jsonl"))
        if gist_jsonl_files:
            # 按文件名排序，确保加载顺序一致
            gist_jsonl_files.sort()
            logger.info(f"找到 {len(gist_jsonl_files)} 个间质瘤数据文件")
            
            for jsonl_file in gist_jsonl_files:
                logger.info(f"加载间质瘤数据文件: {jsonl_file}")
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            if line.strip():  # 跳过空行
                                try:
                                    data = json.loads(line)
                                    # 从answer字段提取真实标签
                                    answer = data.get('answer', '')
                                    true_label = self._extract_label_from_answer(answer)
                                    
                                    test_data.append({
                                        'question': data.get('question', ''),
                                        'true_label': true_label,
                                        'answer': answer,  # 保存原始答案用于参考
                                        'file_name': f"{jsonl_file.name}_{line_num}.jsonl"
                                    })
                                except json.JSONDecodeError as e:
                                    logger.warning(f"跳过文件 {jsonl_file.name} 无效JSON行 {line_num}: {e}")
                except Exception as e:
                    logger.error(f"加载间质瘤数据文件 {jsonl_file} 失败: {e}")
            
            logger.info(f"总共加载了 {len([item for item in test_data if 'gist' in item['file_name']])} 条间质瘤数据")
        
        # 如果没有找到数据，尝试其他可能的文件
        logger.info(f"成功加载 {len(test_data)} 个测试案例")
        print([t['true_label'] for t in test_data])
        return test_data
    
    def _extract_label_from_question(self, question: str) -> str:
        """从问题文本中提取病种标签"""
        question_lower = question.lower()
        
        # 检查关键词来判断疾病类型
        if any(keyword in question for keyword in ['间质瘤', 'gist', '胃肠间质瘤']):
            return 'GIST'
        elif any(keyword in question for keyword in ['平滑肌瘤', '胃平滑肌瘤']):
            return 'LEIOMYOMA'
        elif any(keyword in question for keyword in ['异位胰腺', '胰腺异位', '迷走胰腺']):
            return 'PANCREATIC'
        elif any(keyword in question for keyword in ['神经鞘瘤', '施万细胞瘤']):
            return 'SCHWANNOMA'
        else:
            return 'OTHER'
    
    def _extract_label_from_answer(self, answer: str) -> str:
        """从answer字段中提取病种标签（从## Final Response部分提取）"""
        # 提取## Final Response部分的内容
        final_response_match = re.search(r'## Final Response\s*\n(.*?)(?=\n##|$)', answer, re.DOTALL)
        if final_response_match:
            answer_content = final_response_match.group(1).strip()
        else:
            # 如果没有找到## Final Response，尝试提取<answer>标签（向后兼容）
            answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
            else:
                answer_content = answer.strip()
        
        # 清理答案文本
        # 去除可能的前缀词汇
        answer_content = re.sub(r'^(诊断为?|考虑为?|倾向于?|可能是?|最终诊断为?|结论为?)', '', answer_content)
        answer_content = answer_content.strip()
        
        # 去除可能的标点符号和多余空格
        answer_content = re.sub(r'[，。、；：！？\s]+$', '', answer_content)
        
        # 按优先级检查疾病名称
        if any(keyword in answer_content for keyword in ['胃肠间质瘤', 'GIST', '间质瘤']):
            return 'GIST'
        elif any(keyword in answer_content for keyword in ['平滑肌瘤', '胃平滑肌瘤']):
            return 'LEIOMYOMA'
        elif any(keyword in answer_content for keyword in ['异位胰腺', '胰腺异位', '迷走胰腺']):
            return 'PANCREATIC'
        elif any(keyword in answer_content for keyword in ['神经鞘瘤', '施万细胞瘤']):
            return 'SCHWANNOMA'
        else:
            # 如果直接匹配失败，尝试更宽松的匹配
            disease_patterns = [
                (r'(胃肠间质瘤|GIST|间质瘤)', 'GIST'),
                (r'(平滑肌瘤|胃平滑肌瘤)', 'LEIOMYOMA'),
                (r'(异位胰腺|胰腺异位|迷走胰腺)', 'PANCREATIC'),
                (r'(神经鞘瘤|施万细胞瘤)', 'SCHWANNOMA')
            ]
            
            for pattern, label in disease_patterns:
                if re.search(pattern, answer_content):
                    return label
            
            return 'OTHER'
    
    def evaluate_model(self, model: LocalModel, test_data: List[Dict[str, Any]], max_new_tokens: int = 1000, temperature: float = 0.1) -> EvaluationResult:
        """评估模型性能"""
        logger.info(f"开始评估本地模型: {model.model_name}")
        
        start_time = time.time()
        predictions = []
        
        # 统计每个类别的预测结果
        confusion_matrix = {category: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for category in self.disease_categories.keys()}
        predicted_labels = []
        for i, case in enumerate(test_data):
            try:
                prompt = self.prompt_template.format(question=case['question'])
                prediction = model.generate_response(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                predicted_label = self._extract_label_from_answer(prediction)
                true_label = case['true_label']
                
                predictions.append({
                    'file_name': case['file_name'],
                    'question': case['question'][:100] + "...",  # CSV显示用截断版本
                    'full_question': case['question'],  # 保存完整问题
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'prediction_text': prediction,
                    'correct': predicted_label == true_label
                })
                predicted_labels.append(predicted_label)
                # 更新混淆矩阵
                for category in self.disease_categories.keys():
                    if true_label == category and predicted_label == category:
                        confusion_matrix[category]['tp'] += 1
                    elif true_label == category and predicted_label != category:
                        confusion_matrix[category]['fn'] += 1
                    elif true_label != category and predicted_label == category:
                        confusion_matrix[category]['fp'] += 1
                    elif true_label != category and predicted_label != category:
                        confusion_matrix[category]['tn'] += 1
                
                logger.info(f"案例 {i+1}/{len(test_data)}: {'✓' if predicted_label == true_label else '✗'} ({true_label} -> {predicted_label})")
                time.sleep(0.1)  # 避免请求过快
                
            except Exception as e:
                logger.error(f"评估案例 {i+1} 失败: {e}")
                predictions.append({
                    'file_name': case['file_name'],
                    'question': case['question'][:100] + "...",
                    'full_question': case['question'],
                    'true_label': case['true_label'],
                    'predicted_label': 'ERROR',
                    'prediction_text': f"错误: {e}",
                    'correct': False
                })
        print(predicted_labels)
        execution_time = time.time() - start_time
        
        # 计算各类别指标
        per_class_metrics = {}
        correct_predictions = sum(1 for p in predictions if p['correct'])
        
        for category in self.disease_categories.keys():
            tp = confusion_matrix[category]['tp']
            fp = confusion_matrix[category]['fp']
            fn = confusion_matrix[category]['fn']
            tn = confusion_matrix[category]['tn']
            
            accuracy = (tp + tn) / len(test_data) if len(test_data) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[category] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        
        result = EvaluationResult(
            model_name=model.model_name,
            total_cases=len(test_data),
            correct_predictions=correct_predictions,
            accuracy=correct_predictions / len(test_data) if len(test_data) > 0 else 0,
            per_class_metrics=per_class_metrics,
            execution_time=execution_time,
            cot_examples_used=total_cot_examples_used
        )
        
        self._save_detailed_results(predictions, result)
        return result
    
    def _save_detailed_results(self, predictions: List[Dict], result: EvaluationResult):
        """保存详细结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保results目录存在
        os.makedirs("results", exist_ok=True)
        
        # 保存预测详情为CSV
        predictions_df = pd.DataFrame(predictions)
        predictions_file = f"results/detailed_results_{timestamp}.csv"
        predictions_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
        
        # 保存为JSONL格式
        jsonl_file = f"results/detailed_results_{timestamp}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for pred in predictions:
                # 构造每行的JSON对象
                json_obj = {
                    "file_name": pred['file_name'],
                    "question": pred['full_question'],  # 使用完整问题
                    "true_label": pred['true_label'],
                    "predicted_label": pred['predicted_label'],
                    "model_response": pred['prediction_text'],
                    "correct": pred['correct'],
                    "timestamp": timestamp
                }
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
        
        # 保存评估摘要
        summary = {
            "模型名称": result.model_name,
            "测试案例总数": result.total_cases,
            "正确预测数": result.correct_predictions,
            "总体准确率": f"{result.accuracy:.4f}",
            "执行时间(秒)": f"{result.execution_time:.2f}",
            "评估时间": timestamp,
            "各类别指标": {}
        }
        
        # 添加各类别指标
        disease_names = {
            'GIST': '胃肠间质瘤',
            'LEIOMYOMA': '平滑肌瘤',
            'PANCREATIC': '异位胰腺',
            'SCHWANNOMA': '神经鞘瘤',
            'OTHER': '其他'
        }
        
        for category, metrics in result.per_class_metrics.items():
            category_name = disease_names.get(category, category)
            summary["各类别指标"][category_name] = {
                "准确率": f"{metrics['accuracy']:.4f}",
                "精确率": f"{metrics['precision']:.4f}",
                "召回率": f"{metrics['recall']:.4f}",
                "F1分数": f"{metrics['f1']:.4f}",
                "真阳性": metrics['tp'],
                "假阳性": metrics['fp'],
                "假阴性": metrics['fn'],
                "真阴性": metrics['tn']
            }
        
        summary_file = f"results/evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到: {predictions_file}")
        logger.info(f"JSONL格式结果已保存到: {jsonl_file}")
        logger.info(f"评估摘要已保存到: {summary_file}")
    
    def print_results(self, result: EvaluationResult):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print(f"本地模型评估结果: {result.model_name}")
        print(f"{'='*60}")
        print(f"总体准确率: {result.accuracy:.4f}")
        print(f"总测试案例: {result.total_cases}")
        print(f"正确预测: {result.correct_predictions}")
        print(f"执行时间: {result.execution_time:.2f}秒")
        
        print(f"\n{'='*60}")
        print("各病种详细指标:")
        print(f"{'='*60}")
        
        disease_names = {
            'GIST': '胃肠间质瘤',
            'LEIOMYOMA': '平滑肌瘤', 
            'PANCREATIC': '异位胰腺',
            'SCHWANNOMA': '神经鞘瘤',
            'OTHER': '其他'
        }
        
        for category, metrics in result.per_class_metrics.items():
            name = disease_names.get(category, category)
            print(f"{name:>8} - 准确率: {metrics['accuracy']:.4f}, 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"{'':>10} (TP:{metrics['tp']}, FP:{metrics['fp']}, FN:{metrics['fn']}, TN:{metrics['tn']})")

    def load_cot_examples(self):
        """从jsonl文件加载CoT示例并按病种分类"""
        logger.info(f"正在加载CoT示例文件: {self.cot_file_path}")
        
        if not os.path.exists(self.cot_file_path):
            logger.error(f"CoT文件不存在: {self.cot_file_path}")
            return
        
        try:
            with open(self.cot_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        cot_data = json.loads(line.strip())
                        
                        # 从disease_type字段获取病种
                        disease_type = cot_data.get('disease_type', '')
                        if disease_type in self.disease_mapping:
                            category = self.disease_mapping[disease_type]
                            
                            if category not in self.cot_examples:
                                self.cot_examples[category] = []
                            
                            self.cot_examples[category].append({
                                'question': cot_data.get('question', ''),
                                'think': cot_data.get('think', ''),
                                'answer': cot_data.get('answer', ''),
                                'disease_type': disease_type
                            })
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"解析第{line_num}行JSON失败: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"加载CoT示例失败: {e}")
            return
        
        # 打印加载统计
        total_examples = sum(len(examples) for examples in self.cot_examples.values())
        logger.info(f"成功加载 {total_examples} 个CoT示例:")
        for category, examples in self.cot_examples.items():
            if examples:
                logger.info(f"  {category}: {len(examples)} 个示例")

    def get_random_cot_examples(self, num_examples: int = 2, sampling_method: str = 'proportional') -> List[Dict[str, str]]:
        """从所有病种中获取CoT示例"""
        if sampling_method == 'proportional':
            # 按比例从各个病种中抽取示例
            return self._get_proportional_cot_examples(num_examples)
        elif sampling_method == 'interval':
            # 隔一个选一个的方式
            all_examples = []
            for category, examples in self.cot_examples.items():
                all_examples.extend(examples)
            
            if not all_examples:
                return []
                
            if len(all_examples) < num_examples:
                return all_examples
            
            # 计算间隔步长
            step = max(1, len(all_examples) // num_examples)
            selected_examples = []
            
            # 从随机起始位置开始，按间隔选择
            start_idx = random.randint(0, step - 1) if step > 1 else 0
            
            for i in range(num_examples):
                idx = (start_idx + i * step) % len(all_examples)
                selected_examples.append(all_examples[idx])
                
            return selected_examples
        else:
            # 原来的随机选择方式
            all_examples = []
            for category, examples in self.cot_examples.items():
                all_examples.extend(examples)
            
            if not all_examples:
                return []
            return random.sample(all_examples, min(num_examples, len(all_examples)))

    def _get_proportional_cot_examples(self, num_examples: int) -> List[Dict[str, str]]:
        """按比例从各个病种中抽取CoT示例"""
        if num_examples <= 0:
            return []
        
        # 过滤掉空的类别
        available_categories = {k: v for k, v in self.cot_examples.items() if v}
        
        if not available_categories:
            return []
        
        selected_examples = []
        
        # 计算总的示例数量
        total_examples = sum(len(examples) for examples in available_categories.values())
        
        if num_examples >= len(available_categories):
            # 如果需要的示例数 >= 类别数，先保证每个类别至少选1个
            examples_per_category = {}
            remaining_examples = num_examples
            
            # 第一轮：每个类别至少分配1个
            for category in available_categories:
                examples_per_category[category] = 1
                remaining_examples -= 1
            
            # 第二轮：按比例分配剩余的示例
            if remaining_examples > 0:
                for category, examples in available_categories.items():
                    # 计算该类别应该分配的额外示例数（按比例）
                    category_ratio = len(examples) / total_examples
                    additional_examples = int(remaining_examples * category_ratio)
                    examples_per_category[category] += additional_examples
            
            # 第三轮：处理由于取整导致的差异
            current_total = sum(examples_per_category.values())
            if current_total < num_examples:
                # 还有剩余，按类别示例数量降序补充
                sorted_categories = sorted(available_categories.keys(), 
                                         key=lambda k: len(available_categories[k]), 
                                         reverse=True)
                for category in sorted_categories:
                    if current_total >= num_examples:
                        break
                    if examples_per_category[category] < len(available_categories[category]):
                        examples_per_category[category] += 1
                        current_total += 1
        else:
            # 如果需要的示例数 < 类别数，按比例分配，可能有些类别分不到
            examples_per_category = {}
            for category, examples in available_categories.items():
                category_ratio = len(examples) / total_examples
                allocated = max(0, int(num_examples * category_ratio))
                if allocated > 0:
                    examples_per_category[category] = min(allocated, len(examples))
            
            # 确保总数不超过要求
            current_total = sum(examples_per_category.values())
            if current_total < num_examples:
                # 还有剩余，优先分配给样本多的类别
                sorted_categories = sorted(available_categories.keys(), 
                                         key=lambda k: len(available_categories[k]), 
                                         reverse=True)
                for category in sorted_categories:
                    if current_total >= num_examples:
                        break
                    if category not in examples_per_category:
                        examples_per_category[category] = 1
                        current_total += 1
                    elif examples_per_category[category] < len(available_categories[category]):
                        examples_per_category[category] += 1
                        current_total += 1
        
        # 从每个类别中随机选择指定数量的示例
        for category, count in examples_per_category.items():
            if count > 0:
                category_examples = available_categories[category]
                selected = random.sample(category_examples, min(count, len(category_examples)))
                selected_examples.extend(selected)
        
        # 如果还是不够，从所有示例中随机补充
        if len(selected_examples) < num_examples:
            all_examples = []
            for examples in available_categories.values():
                all_examples.extend(examples)
            
            # 去除已选择的示例
            remaining_examples = [ex for ex in all_examples if ex not in selected_examples]
            
            if remaining_examples:
                additional_needed = num_examples - len(selected_examples)
                additional = random.sample(remaining_examples, 
                                         min(additional_needed, len(remaining_examples)))
                selected_examples.extend(additional)
        
        # 打乱顺序避免类别聚集
        random.shuffle(selected_examples)
        
        return selected_examples[:num_examples]

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于CoT上下文的大模型多病种诊断准确率测评系统")
    parser.add_argument('--model_name', type=str, required=True,
                        help="模型路径或名称")
    parser.add_argument('--port', type=int, default=8168,
                        help="模型服务端口号")
    parser.add_argument('--data_dir', type=str, default="evaluation/data",
                        help="测试数据目录")
    parser.add_argument('--cot_file_path', type=str, 
                        default="../data/train_data/cot_data_2025_7_7/jsonl/cot_data_2025_7_7_20250723_155135_all_diseases_final.jsonl",
                        help="CoT示例文件路径")
    parser.add_argument('--num_cot_examples', type=int, default=0,
                        help="每个案例使用的CoT示例数量 (0表示不使用CoT)")
    parser.add_argument('--sampling_method', type=str, default='proportional',
                        choices=['proportional', 'interval', 'random'],
                        help="CoT示例采样方法: proportional(按比例), interval(间隔), random(随机)")
    parser.add_argument('--max_new_tokens', type=int, default=1000,
                        help="最大生成token数")
    parser.add_argument('--temperature', type=float, default=0.0,
                        help="生成温度")
    parser.add_argument('--use_chat_template', type=bool, default=True,
                        help="是否使用chat template")
    
    args = parser.parse_args()
    
    print("="*60)
    print("大模型多病种诊断准确率测评系统")
    print("支持：胃肠间质瘤、平滑肌瘤、异位胰腺、神经鞘瘤")
    print(f"模型: {args.model_name}")
    print(f"端口: {args.port}")
    print(f"CoT示例数: {args.num_cot_examples}")
    print(f"采样方法: {args.sampling_method}")
    print("="*60)
    
    # 初始化评估器和模型
    evaluator = DiagnosisEvaluator(data_dir=args.data_dir, cot_file_path=args.cot_file_path)
    model = LocalModel(
        model_name=args.model_name,
        port=args.port,
        use_chat_template=args.use_chat_template
    )
    
    # 检查CoT示例是否加载成功
    if args.num_cot_examples > 0 and not evaluator.cot_examples:
        logger.error("没有加载到CoT示例，请检查CoT数据文件")
        return
    
    # 加载测试数据
    test_data = evaluator.load_test_data()
    if not test_data:
        logger.error("没有找到测试数据，请检查数据目录")
        return
    
    print(f"加载了 {len(test_data)} 个测试案例")
    
    # 统计各类别分布
    class_counts = {}
    for case in test_data:
        label = case['true_label']
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print("\n测试数据分布:")
    disease_names = {
        'GIST': '胃肠间质瘤',
        'LEIOMYOMA': '平滑肌瘤',
        'PANCREATIC': '异位胰腺', 
        'SCHWANNOMA': '神经鞘瘤',
        'OTHER': '其他'
    }
    for label, count in class_counts.items():
        name = disease_names.get(label, label)
        print(f"  {name}: {count} 例")
    
    # 显示CoT示例统计
    if args.num_cot_examples > 0:
        print(f"\nCoT示例分布:")
        total_cot_examples = sum(len(examples) for examples in evaluator.cot_examples.values())
        print(f"  总CoT示例数: {total_cot_examples}")
        for category, examples in evaluator.cot_examples.items():
            if examples:
                name = disease_names.get(category, category)
                print(f"  {name}: {len(examples)} 个示例")
    
    # 开始评估
    try:
        result = evaluator.evaluate_model_with_cot(
            model, 
            test_data, 
            num_cot_examples=args.num_cot_examples,
            sampling_method=args.sampling_method,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        evaluator.print_results(result)
        
    except Exception as e:
        logger.error(f"评估失败: {e}")

if __name__ == "__main__":
    main() 
