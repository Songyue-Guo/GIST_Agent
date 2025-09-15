#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于CoT上下文的大模型间质瘤等病种诊断准确率测评脚本
通过读取CoT jsonl文件作为few-shot learning上下文来提升模型表现
支持间质瘤、平滑肌瘤、异位胰腺、神经鞘瘤等多分类评估
"""
import sys
from pathlib import Path
# 添加父目录到Python路径以便导入model包
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import os
import re
import time
import pandas as pd
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
from openai import OpenAI
import openai
from transformers import AutoTokenizer
from jinja2 import Template
# 解决导入路径问题
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model.openrouter import ClaudeModel, GeminiModel

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureModel:
    """Azure OpenAI模型类"""
    def __init__(self, api_key: str = 'sk-qIEoP1B6ZEMVa4sS107e5a155c654187936283F64241Cf8c', 
                 endpoint: str ='https://mdi.hkust-gz.edu.cn/llm_api/v1', 
                 model_name: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
        
    def generate_response(self, prompt: str) -> str:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.endpoint
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )   
        return response.choices[0].message.content or ""

class LocalModel:
    """本地部署模型类（通过OpenAI兼容API调用）"""
    def __init__(self, model_name: str, port: int = 8168, use_chat_template: bool = True):
        self.model_name = model_name
        self.port = port
        self.use_chat_template = use_chat_template
        
        # 初始化API客户端
        self.client = openai.Client(
            base_url=f"http://10.120.20.173:{port}/v1",
            api_key="token-abc123",
        )
        # self.client = openai.Client(
        #     base_url=f"http://10.120.20.168:{port}/v1", 
        #     api_key="token-abc123"
        # )
        
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

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    model_name: str
    total_cases: int
    correct_predictions: int
    accuracy: float
    per_class_metrics: Dict[str, Dict[str, float]]  # 每个类别的precision, recall, f1
    execution_time: float
    cot_examples_used: int

class CoTDiagnosisEvaluator:
    """基于CoT上下文的多病种诊断准确率评估器"""
    
    def __init__(self, data_dir: str = "../data/train_data", 
                 cot_file_path: str = "../data/train_data/cot_data_2025_7_7/jsonl/cot_data_2025_7_7_20250723_155135_all_diseases_final.jsonl"):
        self.data_dir = Path(data_dir)
        self.cot_file_path = cot_file_path
        self.cot_examples = {}  # 按病种分类的CoT示例
        
        # 定义病种类别
        self.disease_categories = {
            'GIST': ['间质瘤', 'GIST', '胃肠间质瘤'],
            'LEIOMYOMA': ['平滑肌瘤', '胃平滑肌瘤'],
            'PANCREATIC': ['异位胰腺', '胰腺异位', '迷走胰腺'],
            'SCHWANNOMA': ['神经鞘瘤'],
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
    
    def create_cot_prompt_template(self, num_examples: int = 2, language: str = "cn") -> str:
        """创建包含CoT示例的诊断提示模板"""
        
        # 从所有病种中随机获取CoT示例
        cot_examples = self.get_random_cot_examples(num_examples)
        
        # 基础模板 问题明确，
        base_template_cn = """你是一位经验丰富的消化内科专家，专门从事胃肠道肿瘤的诊断。请根据以下患者的临床资料，给出你的诊断结论。

请参考提供的参考示例按照以下格式输出：
## Thinking
[请在这里进行详细的临床分析，包括症状分析、影像学特征分析、鉴别诊断等]

## Final Response
[请在这里给出明确的诊断结论，仅包含疾病名称，如：异位胰腺、神经鞘瘤、胃肠间质瘤、平滑肌瘤等]

"""

        base_template_en = """You are a seasoned gastrointestinal oncologist specializing in the diagnosis of gastrointestinal tumors. Please provide your diagnosis based on the clinical information of the following patient.

Please follow the format provided below:
## Thinking
[Please provide a detailed clinical analysis, including symptom analysis, imaging features, and differential diagnosis.]

## Final Response
[Please provide a clear diagnosis conclusion, only including the disease name, such as: 异位胰腺、神经鞘瘤、胃肠间质瘤、平滑肌瘤等]

"""     
        if language == "cn":
            base_template = base_template_cn
        else:
            base_template = base_template_en
            
        # 添加CoT示例
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
        
        base_template += "**当前需要诊断的患者情况：**\n{question}\n"
        
        return base_template
    
    def load_test_data(self, target_disease: Optional[str] = None, max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
        """加载测试数据
        
        Args:
            target_disease: 目标疾病类型 ('GIST', 'LEIOMYOMA', 'PANCREATIC', 'SCHWANNOMA', 'ALL')
            max_cases: 最大案例数量限制
        """
        test_data = []
        
        # 从间质瘤数据文件夹加载
        gist_data_dir = self.data_dir / "cot_data_gist"
        if gist_data_dir.exists():
            # 找最新的final文件
            final_files = list(gist_data_dir.glob("*final.json"))
            if final_files:
                # 按修改时间排序，取最新的
                latest_file = max(final_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"加载间质瘤数据文件: {latest_file}")
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data_list = json.load(f)
                        for data in data_list:
                            test_data.append({
                                'question': data.get('question', ''),
                                'true_label': 'GIST',  # 间质瘤数据标签为GIST
                                'file_name': f"gist_{len(test_data)}.json"
                            })
                except Exception as e:
                    logger.error(f"加载间质瘤数据失败: {e}")
        
        # 从非间质瘤数据文件夹加载
        n_gist_data_dir = self.data_dir / "cot_data_n_gist"
        if n_gist_data_dir.exists():
            # 找最新的final文件
            final_files = list(n_gist_data_dir.glob("*final.json"))
            if final_files:
                # 按修改时间排序，取最新的
                latest_file = max(final_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"加载非间质瘤数据文件: {latest_file}")
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data_list = json.load(f)
                        for data in data_list:
                            # 从disease_type字段或答案中提取标签
                            disease_type = data.get('disease_type', '')
                            answer = data.get('answer', '')
                            true_label = self._extract_label_from_disease_type(disease_type, answer)
                            
                            test_data.append({
                                'question': data.get('question', ''),
                                'true_label': true_label,
                                'file_name': f"n_gist_{len(test_data)}.json"
                            })
                except Exception as e:
                    logger.error(f"加载非间质瘤数据失败: {e}")
        
        logger.info(f"成功加载 {len(test_data)} 个测试案例")
        
        # 应用疾病类型过滤
        if target_disease and target_disease != 'ALL':
            if target_disease in self.disease_categories:
                filtered_data = [case for case in test_data if case['true_label'] == target_disease]
                logger.info(f"过滤后保留 {target_disease} 类型案例: {len(filtered_data)} 个")
                test_data = filtered_data
            else:
                logger.warning(f"无效的疾病类型: {target_disease}，支持的类型: {list(self.disease_categories.keys())}")
        
        # 应用数量限制
        if max_cases and max_cases > 0:
            if len(test_data) > max_cases:
                # 随机采样以保持数据分布
                import random
                random.shuffle(test_data)
                test_data = test_data[:max_cases]
                logger.info(f"限制案例数量为: {max_cases} 个")
        
        logger.info(f"最终测试案例数量: {len(test_data)} 个")
        return test_data
    
    def _extract_label_from_disease_type(self, disease_type: str, answer: str) -> str:
        """从disease_type字段或答案中提取病种标签"""
        # 首先检查disease_type字段
        if disease_type:
            if '神经鞘瘤' in disease_type:
                return 'SCHWANNOMA'
            elif '异位胰腺' in disease_type:
                return 'PANCREATIC'
            elif '平滑肌瘤' in disease_type:
                return 'LEIOMYOMA'
            elif '胃间质瘤' in disease_type or '间质瘤' in disease_type:
                return 'GIST'
        
        # 如果disease_type为空，从答案中提取
        answer_clean = re.sub(r'<[^>]+>', '', answer).strip()
        
        # 按优先级检查各种疾病关键词
        for disease_type, keywords in self.disease_categories.items():
            if disease_type == 'OTHER':
                continue
            for keyword in keywords:
                if keyword in answer_clean:
                    return disease_type
        
        return 'OTHER'
    
    def _extract_label_from_answer(self, answer: str) -> str:
        """从答案中提取病种标签（用于模型预测结果）"""
        # 清理答案文本
        answer_clean = re.sub(r'<[^>]+>', '', answer).strip()
        
        # 按优先级检查各种疾病关键词
        for disease_type, keywords in self.disease_categories.items():
            if disease_type == 'OTHER':
                continue
            for keyword in keywords:
                if keyword in answer_clean:
                    return disease_type
        
        return 'OTHER'
    
    def evaluate_model_with_cot(self, model, test_data: List[Dict[str, Any]], 
                               num_cot_examples: int = 2) -> EvaluationResult:
        """使用CoT上下文评估模型性能"""
        logger.info(f"开始评估 {model.model_name} 模型（使用CoT上下文，每个案例{num_cot_examples}个随机示例）")
        
        start_time = time.time()
        predictions = []
        total_cot_examples_used = 0
        
        # 统计每个类别的预测结果
        confusion_matrix = {category: {'tp': 0, 'fp': 0, 'fn': 0} for category in self.disease_categories.keys()}
        
        for i, case in enumerate(test_data):
            try:
                true_label = case['true_label']
                
                # 为当前案例创建包含随机CoT示例的prompt
                prompt_template = self.create_cot_prompt_template(num_cot_examples)
                prompt = prompt_template.format(question=case['question'])
                
                # 统计使用的CoT示例数量
                used_examples = len(self.get_random_cot_examples(num_cot_examples))
                total_cot_examples_used += used_examples
                
                prediction = model.generate_response(prompt)
                predicted_label = self._extract_label_from_answer(prediction)
                
                predictions.append({
                    'file_name': case['file_name'],
                    'question': case['question'][:100] + "...",  # CSV显示用截断版本
                    'full_question': case['question'],  # 保存完整问题
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'prediction_text': prediction,
                    'correct': predicted_label == true_label,
                    'cot_examples_used': used_examples
                })
                
                # 更新混淆矩阵
                for category in self.disease_categories.keys():
                    if true_label == category and predicted_label == category:
                        confusion_matrix[category]['tp'] += 1
                    elif true_label == category and predicted_label != category:
                        confusion_matrix[category]['fn'] += 1
                    elif true_label != category and predicted_label == category:
                        confusion_matrix[category]['fp'] += 1
                
                logger.info(f"案例 {i+1}/{len(test_data)}: {'✓' if predicted_label == true_label else '✗'} "
                          f"({true_label} -> {predicted_label}, CoT示例: {used_examples})")
                time.sleep(0.5)  # 避免API限制
                
            except Exception as e:
                logger.error(f"评估案例 {i+1} 失败: {e}")
                predictions.append({
                    'file_name': case['file_name'],
                    'question': case['question'][:100] + "...",
                    'full_question': case['question'],
                    'true_label': case['true_label'],
                    'predicted_label': 'ERROR',
                    'prediction_text': f"错误: {e}",
                    'correct': False,
                    'cot_examples_used': 0
                })
        
        execution_time = time.time() - start_time
        
        # 计算各类别指标
        per_class_metrics = {}
        correct_predictions = sum(1 for p in predictions if p['correct'])
        total_samples = len(test_data)
        
        for category in self.disease_categories.keys():
            tp = confusion_matrix[category]['tp']
            fp = confusion_matrix[category]['fp']
            fn = confusion_matrix[category]['fn']
            tn = total_samples - tp - fp - fn  # 真负例
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total_samples if total_samples > 0 else 0  # 每个类别的准确率
            
            per_class_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
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
    
    def evaluate_single_case(self, model, case_data: Dict[str, Any], 
                           num_cot_examples: int = 2) -> Dict[str, Any]:
        """评估单个案例"""
        logger.info(f"开始评估单个案例: {case_data.get('file_name', 'unknown')}")
        
        try:
            true_label = case_data['true_label']
            
            # 为当前案例创建包含随机CoT示例的prompt
            prompt_template = self.create_cot_prompt_template(num_cot_examples)
            prompt = prompt_template.format(question=case_data['question'])
            
            # 统计使用的CoT示例数量
            used_examples = len(self.get_random_cot_examples(num_cot_examples))
            
            prediction = model.generate_response(prompt)
            predicted_label = self._extract_label_from_answer(prediction)
            
            result = {
                'file_name': case_data['file_name'],
                'question': case_data['question'],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'prediction_text': prediction,
                'correct': predicted_label == true_label,
                'cot_examples_used': used_examples,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"评估结果: {'✓' if predicted_label == true_label else '✗'} "
                      f"({true_label} -> {predicted_label}, CoT示例: {used_examples})")
            
            return result
            
        except Exception as e:
            logger.error(f"评估单个案例失败: {e}")
            return {
                'file_name': case_data.get('file_name', 'unknown'),
                'question': case_data.get('question', '')[:100] + "...",
                'true_label': case_data.get('true_label', 'UNKNOWN'),
                'predicted_label': 'ERROR',
                'prediction_text': f"错误: {e}",
                'correct': False,
                'cot_examples_used': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def find_case_by_id_or_name(self, test_data: List[Dict[str, Any]], 
                               case_id: Optional[str] = None, patient_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据案例ID或患者姓名查找案例"""
        if case_id:
            # 根据file_name查找
            for case in test_data:
                if case_id in case.get('file_name', ''):
                    return case
            logger.error(f"未找到案例ID包含 '{case_id}' 的案例")
        
        if patient_name:
            # 根据问题内容中的患者姓名查找
            for case in test_data:
                if patient_name in case.get('question', ''):
                    return case
            logger.error(f"未找到患者姓名包含 '{patient_name}' 的案例")
        
        logger.error("未提供有效的案例ID或患者姓名")
        return None
    
    def list_available_cases(self, test_data: List[Dict[str, Any]], max_display: int = 20):
        """列出可用的测试案例"""
        logger.info(f"可用测试案例 (共 {len(test_data)} 个):")
        
        # 按疾病类型分组显示
        disease_names = {
            'GIST': '胃肠间质瘤',
            'LEIOMYOMA': '平滑肌瘤',
            'PANCREATIC': '异位胰腺',
            'SCHWANNOMA': '神经鞘瘤',
            'OTHER': '其他'
        }
        
        grouped_cases = {}
        for case in test_data:
            disease = case['true_label']
            if disease not in grouped_cases:
                grouped_cases[disease] = []
            grouped_cases[disease].append(case)
        
        displayed_count = 0
        for disease, cases in grouped_cases.items():
            disease_name = disease_names.get(disease, disease)
            print(f"\n{disease_name} ({len(cases)} 个案例):")
            
            for i, case in enumerate(cases):
                if displayed_count >= max_display:
                    print(f"    ... (还有 {len(test_data) - displayed_count} 个案例未显示)")
                    return
                
                file_name = case.get('file_name', f'case_{i}')
                question_preview = case.get('question', '')[:50] + "..."
                print(f"    {displayed_count + 1:3d}. {file_name} - {question_preview}")
                displayed_count += 1
    
    def _save_detailed_results(self, predictions: List[Dict], result: EvaluationResult):
        """保存详细结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保results目录存在
        os.makedirs("results", exist_ok=True)
        
        # 保存预测详情为CSV
        predictions_df = pd.DataFrame(predictions)
        predictions_file = f"results/cot_detailed_results_{timestamp}.csv"
        predictions_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
        
        # 保存为JSONL格式
        jsonl_file = f"results/cot_detailed_results_{timestamp}.jsonl"
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
                    "cot_examples_used": pred.get('cot_examples_used', 0),
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
            "总CoT示例使用数": result.cot_examples_used,
            "平均每案例CoT示例数": f"{result.cot_examples_used / result.total_cases:.2f}" if result.total_cases > 0 else "0",
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
                "精确率": f"{metrics['precision']:.4f}",
                "召回率": f"{metrics['recall']:.4f}",
                "F1分数": f"{metrics['f1']:.4f}",
                "准确率": f"{metrics['accuracy']:.4f}",
                "真阳性": metrics['tp'],
                "假阳性": metrics['fp'],
                "假阴性": metrics['fn'],
                "真阴性": metrics['tn']
            }
        
        summary_file = f"results/cot_evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到: {predictions_file}")
        logger.info(f"JSONL格式结果已保存到: {jsonl_file}")
        logger.info(f"评估摘要已保存到: {summary_file}")
    
    def print_results(self, result: EvaluationResult):
        """打印评估结果"""
        print(f"\n{'='*60}")
        print(f"基于CoT上下文的Azure OpenAI 模型评估结果")
        print(f"{'='*60}")
        print(f"总体准确率: {result.accuracy:.4f}")
        print(f"总测试案例: {result.total_cases}")
        print(f"正确预测: {result.correct_predictions}")
        print(f"总CoT示例使用数: {result.cot_examples_used}")
        print(f"平均每案例CoT示例数: {result.cot_examples_used / result.total_cases:.2f}")
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
            print(f"{name:>8} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, 准确率: {metrics['accuracy']:.4f}")
            print(f"{'':>10} (TP:{metrics['tp']}, FP:{metrics['fp']}, FN:{metrics['fn']}, TN:{metrics['tn']})")

def main():
    """主函数"""

    parser = argparse.ArgumentParser(description="基于CoT上下文的大模型多病种诊断准确率测评系统")
    parser.add_argument('--model_name', type=str, required=True, default="anthropic/claude-sonnet-4",
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
    parser.add_argument('--target_disease', type=str, default=None,
                        choices=['GIST', 'LEIOMYOMA', 'PANCREATIC', 'SCHWANNOMA', 'ALL'],
                        help="目标疾病类型 (GIST=间质瘤, LEIOMYOMA=平滑肌瘤, PANCREATIC=异位胰腺, SCHWANNOMA=神经鞘瘤, ALL=所有)")
    parser.add_argument('--max_cases', type=int, default=None,
                        help="最大测试案例数量限制")
    parser.add_argument('--language', type=str, default="cn",
                        choices=['cn', 'en'],
                        help="语言")
    parser.add_argument('--mode', type=str, default="batch",
                        choices=['batch', 'single', 'interactive'],
                        help="运行模式: batch(批量处理), single(单个案例), interactive(交互模式)")
    parser.add_argument('--case_id', type=str, default=None,
                        help="指定案例ID (仅在single模式下使用)")
    parser.add_argument('--patient_name', type=str, default=None,
                        help="指定患者姓名 (仅在single模式下使用)")
    
    args = parser.parse_args()

    print("="*60)
    print(f"基于CoT上下文的大模型多病种诊断准确率测评系统")
    print("支持：胃肠间质瘤、平滑肌瘤、异位胰腺、神经鞘瘤")
    print(f"运行模式: {args.mode}")
    print(f"模型: {args.model_name}")
    if args.target_disease:
        disease_names = {
            'GIST': '胃肠间质瘤',
            'LEIOMYOMA': '平滑肌瘤',
            'PANCREATIC': '异位胰腺',
            'SCHWANNOMA': '神经鞘瘤',
            'ALL': '所有疾病'
        }
        print(f"目标疾病: {disease_names.get(args.target_disease, args.target_disease)}")
    if args.max_cases:
        print(f"案例限制: {args.max_cases} 个")
    if args.case_id:
        print(f"指定案例ID: {args.case_id}")
    if args.patient_name:
        print(f"指定患者姓名: {args.patient_name}")
    print(f"CoT示例数: {args.num_cot_examples}")
    print(f"采样方法: {args.sampling_method}")
    print("="*60)
    
    # 初始化评估器和模型
    evaluator = CoTDiagnosisEvaluator()
    # according to the model_name, choose the model
    if "claude" in args.model_name:
        model = ClaudeModel(model_name = args.model_name)
    elif "gemini" in args.model_name:
        model = GeminiModel(model_name = args.model_name)
    elif "gpt" in args.model_name:
        model = AzureModel(model_name = args.model_name)
    else:
        model = LocalModel(model_name = args.model_name)  # 使用默认配置
    
    # 检查CoT示例是否加载成功
    if not evaluator.cot_examples:
        logger.error("没有加载到CoT示例，请检查CoT数据文件")
        return
    
    # 加载测试数据
    test_data = evaluator.load_test_data(target_disease=args.target_disease, max_cases=args.max_cases)
    if not test_data:
        logger.error("没有找到测试数据，请检查数据目录")
        return
    
    print(f"加载了 {len(test_data)} 个测试案例")
    if args.target_disease:
        print(f"目标疾病类型: {args.target_disease}")
    if args.max_cases:
        print(f"案例数量限制: {args.max_cases}")
    
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
    print(f"\nCoT示例分布:")
    total_cot_examples = sum(len(examples) for examples in evaluator.cot_examples.values())
    print(f"  总CoT示例数: {total_cot_examples}")
    for category, examples in evaluator.cot_examples.items():
        if examples:
            name = disease_names.get(category, category)
            print(f"  {name}: {len(examples)} 个示例")
    
    # 根据模式执行相应操作
    try:
        if args.mode == "batch":
            # 批量评估模式
            result = evaluator.evaluate_model_with_cot(
                model, test_data, 
                num_cot_examples=args.num_cot_examples
            )
            evaluator.print_results(result)
            
        elif args.mode == "single":
            # 单个案例评估模式
            if not args.case_id and not args.patient_name:
                logger.error("单个案例模式需要指定 --case_id 或 --patient_name 参数")
                return 1
            
            target_case = evaluator.find_case_by_id_or_name(
                test_data, args.case_id, args.patient_name
            )
            
            if not target_case:
                logger.error("未找到指定的案例")
                evaluator.list_available_cases(test_data)
                return 1
            
            logger.info(f"开始评估指定案例: {target_case['file_name']}")
            result = evaluator.evaluate_single_case(
                model, target_case, 
                num_cot_examples=args.num_cot_examples
            )
            
            # 显示单个案例结果
            print_single_case_result(result)
            
        elif args.mode == "interactive":
            # 交互模式
            return run_interactive_evaluation(evaluator, model, test_data, args)
            
    except Exception as e:
        logger.error(f"评估失败: {e}")
        return 1
    
    return 0


def print_single_case_result(result: Dict[str, Any]):
    """打印单个案例的评估结果"""
    print(f"\n{'='*60}")
    print(f"单个案例评估结果")
    print(f"{'='*60}")
    print(f"案例文件: {result['file_name']}")
    print(f"真实标签: {result['true_label']}")
    print(f"预测标签: {result['predicted_label']}")
    print(f"预测正确: {'✓' if result['correct'] else '✗'}")
    print(f"CoT示例使用数: {result['cot_examples_used']}")
    print(f"评估时间: {result['timestamp']}")
    print(f"\n问题内容:")
    print(f"{result['question'][:200]}...")
    print(f"\n模型回答:")
    print(f"{result['prediction_text']}")


def run_interactive_evaluation(evaluator, model, test_data: List[Dict[str, Any]], args) -> int:
    """运行交互式评估模式"""
    print(f"\n{'='*60}")
    print("交互式评估模式")
    print(f"{'='*60}")
    
    while True:
        print("\n请选择操作:")
        print("1. 查看所有可用案例")
        print("2. 评估指定案例")
        print("3. 批量评估所有案例")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            # 列出所有可用案例
            evaluator.list_available_cases(test_data)
            
        elif choice == "2":
            # 评估指定案例
            case_identifier = input("\n请输入案例ID或患者姓名关键词: ").strip()
            if not case_identifier:
                print("案例标识符不能为空")
                continue
            
            target_case = evaluator.find_case_by_id_or_name(
                test_data, case_identifier, case_identifier
            )
            
            if not target_case:
                print("未找到指定的案例")
                continue
            
            print(f"\n开始评估案例: {target_case['file_name']}")
            result = evaluator.evaluate_single_case(
                model, target_case, 
                num_cot_examples=args.num_cot_examples
            )
            print_single_case_result(result)
            
        elif choice == "3":
            # 批量评估
            confirm = input("\n确认要批量评估所有案例吗？(y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                result = evaluator.evaluate_model_with_cot(
                    model, test_data, 
                    num_cot_examples=args.num_cot_examples
                )
                evaluator.print_results(result)
            else:
                print("已取消批量评估")
                
        elif choice == "4":
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入")
    
    return 0


if __name__ == "__main__":
    exit(main()) 