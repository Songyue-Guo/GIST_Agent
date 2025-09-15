#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型间质瘤等病种诊断准确率测评脚本
支持间质瘤、平滑肌瘤、异位胰腺、神经鞘瘤等多分类评估
"""

import json
import os
import re
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from openai import AzureOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureModel:
    """Azure OpenAI模型类"""
    def __init__(self, api_key: str = 'dbfa63b54a2744d7aba5c2008b125a86', 
                 endpoint: str ='https://mdi-gpt-4o.openai.azure.com/', 
                 model_name: str = "gpt-4o-global"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
        
    def generate_response(self, prompt: str) -> str:
        client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-02-01",
            azure_endpoint=self.endpoint,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )   
        return response.choices[0].message.content or ""

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    model_name: str
    total_cases: int
    correct_predictions: int
    accuracy: float
    per_class_metrics: Dict[str, Dict[str, float]]  # 每个类别的precision, recall, f1
    execution_time: float

class DiagnosisEvaluator:
    """多病种诊断准确率评估器"""
    
    def __init__(self, data_dir: str = "../data/train_data"):
        self.data_dir = Path(data_dir)
        self.prompt_template = self._create_prompt_template()
        # 定义病种类别
        self.disease_categories = {
            'GIST': ['间质瘤', 'GIST', '胃肠间质瘤'],
            'LEIOMYOMA': ['平滑肌瘤', '胃平滑肌瘤'],
            'PANCREATIC': ['异位胰腺', '胰腺异位', '迷走胰腺'],
            'SCHWANNOMA': ['神经鞘瘤', '施万细胞瘤'],
            'OTHER': []  # 其他疾病
        }
        
    def _create_prompt_template(self) -> str:
        """创建诊断提示模板"""
        template = """
                你是一位经验丰富的消化内科专家，专门从事胃肠道肿瘤的诊断。请根据以下患者的临床资料，给出你的诊断结论。
                重要说明：
                1. 请仔细分析患者的症状、体征、影像学检查和内镜检查结果
                2. 请给出明确的诊断结论，格式为："<answer>[具体疾病名称]</answer>"
                3. 输出仅包含诊断结论，格式为："<answer>[具体疾病名称]</answer>"，不要包含其他任何内容，不要包含任何解释
                患者临床资料：
                {question}
                """
        return template
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
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
    
    def evaluate_model(self, model: AzureModel, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """评估模型性能"""
        logger.info(f"开始评估Azure OpenAI模型")
        
        start_time = time.time()
        predictions = []
        
        # 统计每个类别的预测结果
        confusion_matrix = {category: {'tp': 0, 'fp': 0, 'fn': 0} for category in self.disease_categories.keys()}
        
        for i, case in enumerate(test_data):
            try:
                prompt = self.prompt_template.format(question=case['question'])
                prediction = model.generate_response(prompt)
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
                
                # 更新混淆矩阵
                for category in self.disease_categories.keys():
                    if true_label == category and predicted_label == category:
                        confusion_matrix[category]['tp'] += 1
                    elif true_label == category and predicted_label != category:
                        confusion_matrix[category]['fn'] += 1
                    elif true_label != category and predicted_label == category:
                        confusion_matrix[category]['fp'] += 1
                
                logger.info(f"案例 {i+1}/{len(test_data)}: {'✓' if predicted_label == true_label else '✗'} ({true_label} -> {predicted_label})")
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
                    'correct': False
                })
        
        execution_time = time.time() - start_time
        
        # 计算各类别指标
        per_class_metrics = {}
        correct_predictions = sum(1 for p in predictions if p['correct'])
        
        for category in self.disease_categories.keys():
            tp = confusion_matrix[category]['tp']
            fp = confusion_matrix[category]['fp']
            fn = confusion_matrix[category]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        result = EvaluationResult(
            model_name="GPT-4o",
            total_cases=len(test_data),
            correct_predictions=correct_predictions,
            accuracy=correct_predictions / len(test_data) if len(test_data) > 0 else 0,
            per_class_metrics=per_class_metrics,
            execution_time=execution_time
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
                "精确率": f"{metrics['precision']:.4f}",
                "召回率": f"{metrics['recall']:.4f}",
                "F1分数": f"{metrics['f1']:.4f}",
                "真阳性": metrics['tp'],
                "假阳性": metrics['fp'],
                "假阴性": metrics['fn']
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
        print(f"Azure OpenAI 模型评估结果")
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
            print(f"{name:>8} - 精确率: {metrics['precision']:.4f}, 召回率: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            print(f"{'':>10} (TP:{metrics['tp']}, FP:{metrics['fp']}, FN:{metrics['fn']})")

def main():
    """主函数"""
    print("="*60)
    print("大模型多病种诊断准确率测评系统")
    print("支持：胃肠间质瘤、平滑肌瘤、异位胰腺、神经鞘瘤")
    print("="*60)
    
    # 初始化评估器和模型
    evaluator = DiagnosisEvaluator()
    model = AzureModel()  # 使用默认配置
    
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
    
    # 开始评估
    try:
        result = evaluator.evaluate_model(model, test_data)
        evaluator.print_results(result)
        
    except Exception as e:
        logger.error(f"评估失败: {e}")

if __name__ == "__main__":
    main() 