#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型间质瘤诊断准确率测评脚本
用于评估各种大模型（Azure GPT, Claude等）在间质瘤和非间质瘤诊断上的准确率
"""

import json
import os
import re
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """评估结果数据类"""
    model_name: str
    total_cases: int
    correct_predictions: int
    accuracy: float
    gist_tp: int  # 间质瘤真阳性
    gist_fp: int  # 间质瘤假阳性
    gist_tn: int  # 间质瘤真阴性
    gist_fn: int  # 间质瘤假阴性
    gist_precision: float
    gist_recall: float
    gist_f1: float
    non_gist_precision: float
    non_gist_recall: float
    non_gist_f1: float
    execution_time: float

class LLMInterface(ABC):
    """大模型接口抽象类"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """生成回复"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称"""
        pass

class AzureGPTModel(LLMInterface):
    """Azure GPT模型实现"""
    
    def __init__(self, api_key: str, endpoint: str, model_name: str = "gpt-4"):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
        # 这里需要根据实际的Azure API进行实现
        
    def generate_response(self, prompt: str) -> str:
        """调用Azure GPT API生成回复"""
        try:
            # 模拟API调用 - 实际使用时需要替换为真实的Azure API调用
            from openai import AzureOpenAI
            # openai.api_type = "azure"
            # openai.api_base = self.endpoint
            # openai.api_version = "2023-05-15"
            # openai.api_key = self.api_key
            # 创建AzureOpenAI客户端
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2023-05-15",
                azure_endpoint=self.endpoint,
            )
            response = client.chat.completions.create(
                model =self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content or ""
            # return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Azure GPT API调用失败: {e}")
            return ""
    
    def get_model_name(self) -> str:
        return f"Azure-{self.model_name}"

class MockLLMModel(LLMInterface):
    """模拟大模型实现（用于测试）"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    def generate_response(self, prompt: str) -> str:
        """模拟生成回复"""
        # 简单的模拟逻辑，实际使用时会替换为真实模型
        if "间质瘤" in prompt or "GIST" in prompt:
            return "胃肠间质瘤"
        else:
            return "胃腺癌"
    
    def get_model_name(self) -> str:
        return self.model_name

class DiagnosisEvaluator:
    """间质瘤诊断准确率评估器"""
    
    def __init__(self, data_dir: str = "../data/train_data"):
        self.data_dir = Path(data_dir)
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> str:
        """创建诊断提示模板"""
        template = """
你是一位经验丰富的消化内科专家，专门从事胃肠道肿瘤的诊断。请根据以下患者的临床资料，给出你的诊断结论。

重要说明：
1. 请仔细分析患者的症状、体征、影像学检查和内镜检查结果
2. 重点关注是否为胃肠间质瘤（GIST）
3. 请给出明确的诊断结论，格式为："诊断：[具体疾病名称]"
4. 如果是间质瘤，请明确指出；如果不是，请说明具体是什么疾病

患者临床资料：
{question}

请基于上述信息给出你的诊断：
"""
        return template
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        test_data = []
        
        # 加载所有JSON文件
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取真实标签
                    true_label = self._extract_label_from_answer(data.get('answer', ''))
                    test_data.append({
                        'question': data.get('question', ''),
                        'true_answer': data.get('answer', ''),
                        'true_label': true_label,
                        'file_name': json_file.name
                    })
            except Exception as e:
                logger.error(f"加载文件 {json_file} 失败: {e}")
        
        logger.info(f"成功加载 {len(test_data)} 个测试案例")
        return test_data
    
    def _extract_label_from_answer(self, answer: str) -> str:
        """从答案中提取标签（间质瘤 vs 非间质瘤）"""
        # 清理答案文本
        answer_clean = re.sub(r'<[^>]+>', '', answer).strip()
        
        # 检查是否包含间质瘤相关关键词
        gist_keywords = ['间质瘤', 'GIST', '胃肠间质瘤', '间叶源性肿瘤']
        for keyword in gist_keywords:
            if keyword in answer_clean:
                return 'GIST'
        
        return 'NON_GIST'
    
    def _extract_label_from_prediction(self, prediction: str) -> str:
        """从模型预测中提取标签"""
        return self._extract_label_from_answer(prediction)
    
    def evaluate_model(self, model: LLMInterface, test_data: List[Dict[str, Any]]) -> EvaluationResult:
        """评估单个模型"""
        logger.info(f"开始评估模型: {model.get_model_name()}")
        
        start_time = time.time()
        correct_predictions = 0
        predictions = []
        
        # 混淆矩阵计数器
        gist_tp = gist_fp = gist_tn = gist_fn = 0
        
        for i, case in enumerate(test_data):
            try:
                # 生成提示
                prompt = self.prompt_template.format(question=case['question'])
                
                # 获取模型预测
                prediction = model.generate_response(prompt)
                predicted_label = self._extract_label_from_prediction(prediction)
                true_label = case['true_label']
                
                predictions.append({
                    'file_name': case['file_name'],
                    'question': case['question'][:100] + "...",  # 截断显示
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'prediction_text': prediction,
                    'correct': predicted_label == true_label
                })
                
                # 更新混淆矩阵
                if true_label == 'GIST' and predicted_label == 'GIST':
                    gist_tp += 1
                elif true_label == 'GIST' and predicted_label == 'NON_GIST':
                    gist_fn += 1
                elif true_label == 'NON_GIST' and predicted_label == 'GIST':
                    gist_fp += 1
                elif true_label == 'NON_GIST' and predicted_label == 'NON_GIST':
                    gist_tn += 1
                
                if predicted_label == true_label:
                    correct_predictions += 1
                
                logger.info(f"案例 {i+1}/{len(test_data)}: {'✓' if predicted_label == true_label else '✗'}")
                
                # 添加延时避免API限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"评估案例 {i+1} 失败: {e}")
                predictions.append({
                    'file_name': case['file_name'],
                    'question': case['question'][:100] + "...",
                    'true_label': case['true_label'],
                    'predicted_label': 'ERROR',
                    'prediction_text': f"错误: {e}",
                    'correct': False
                })
        
        execution_time = time.time() - start_time
        
        # 计算指标
        total_cases = len(test_data)
        accuracy = correct_predictions / total_cases if total_cases > 0 else 0
        
        # GIST类别的精确率、召回率、F1
        gist_precision = gist_tp / (gist_tp + gist_fp) if (gist_tp + gist_fp) > 0 else 0
        gist_recall = gist_tp / (gist_tp + gist_fn) if (gist_tp + gist_fn) > 0 else 0
        gist_f1 = 2 * gist_precision * gist_recall / (gist_precision + gist_recall) if (gist_precision + gist_recall) > 0 else 0
        
        # NON_GIST类别的精确率、召回率、F1
        non_gist_precision = gist_tn / (gist_tn + gist_fn) if (gist_tn + gist_fn) > 0 else 0
        non_gist_recall = gist_tn / (gist_tn + gist_fp) if (gist_tn + gist_fp) > 0 else 0
        non_gist_f1 = 2 * non_gist_precision * non_gist_recall / (non_gist_precision + non_gist_recall) if (non_gist_precision + non_gist_recall) > 0 else 0
        
        result = EvaluationResult(
            model_name=model.get_model_name(),
            total_cases=total_cases,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            gist_tp=gist_tp,
            gist_fp=gist_fp,
            gist_tn=gist_tn,
            gist_fn=gist_fn,
            gist_precision=gist_precision,
            gist_recall=gist_recall,
            gist_f1=gist_f1,
            non_gist_precision=non_gist_precision,
            non_gist_recall=non_gist_recall,
            non_gist_f1=non_gist_f1,
            execution_time=execution_time
        )
        
        # 保存详细预测结果
        self._save_detailed_results(model.get_model_name(), predictions, result)
        
        return result
    
    def _save_detailed_results(self, model_name: str, predictions: List[Dict], result: EvaluationResult):
        """保存详细的预测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存预测详情
        predictions_df = pd.DataFrame(predictions)
        predictions_file = f"detailed_results_{model_name}_{timestamp}.csv"
        predictions_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
        
        # 保存评估摘要
        summary = {
            "模型名称": result.model_name,
            "测试案例总数": result.total_cases,
            "正确预测数": result.correct_predictions,
            "总体准确率": f"{result.accuracy:.4f}",
            "间质瘤_真阳性": result.gist_tp,
            "间质瘤_假阳性": result.gist_fp,
            "间质瘤_真阴性": result.gist_tn,
            "间质瘤_假阴性": result.gist_fn,
            "间质瘤_精确率": f"{result.gist_precision:.4f}",
            "间质瘤_召回率": f"{result.gist_recall:.4f}",
            "间质瘤_F1分数": f"{result.gist_f1:.4f}",
            "非间质瘤_精确率": f"{result.non_gist_precision:.4f}",
            "非间质瘤_召回率": f"{result.non_gist_recall:.4f}",
            "非间质瘤_F1分数": f"{result.non_gist_f1:.4f}",
            "执行时间(秒)": f"{result.execution_time:.2f}",
            "评估时间": timestamp
        }
        
        summary_file = f"evaluation_summary_{model_name}_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细结果已保存到: {predictions_file}")
        logger.info(f"评估摘要已保存到: {summary_file}")
    
    def compare_models(self, results: List[EvaluationResult]) -> pd.DataFrame:
        """比较多个模型的性能"""
        comparison_data = []
        
        for result in results:
            comparison_data.append({
                "模型名称": result.model_name,
                "总体准确率": f"{result.accuracy:.4f}",
                "间质瘤精确率": f"{result.gist_precision:.4f}",
                "间质瘤召回率": f"{result.gist_recall:.4f}",
                "间质瘤F1": f"{result.gist_f1:.4f}",
                "非间质瘤精确率": f"{result.non_gist_precision:.4f}",
                "非间质瘤召回率": f"{result.non_gist_recall:.4f}",
                "非间质瘤F1": f"{result.non_gist_f1:.4f}",
                "执行时间(秒)": f"{result.execution_time:.2f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"model_comparison_{timestamp}.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"模型比较结果已保存到: {comparison_file}")
        return comparison_df

def main():
    """主函数 - 执行评估流程"""
    # 初始化评估器
    evaluator = DiagnosisEvaluator()
    
    # 加载测试数据
    test_data = evaluator.load_test_data()
    if not test_data:
        logger.error("没有找到测试数据，请检查数据目录")
        return
    
    # 初始化模型列表（实际使用时需要配置真实的API）
    models = [
        MockLLMModel("GPT-4-Mock"),  # 模拟模型，实际使用时替换为真实模型
        MockLLMModel("Claude-3-Mock"),  # 模拟模型
        # AzureGPTModel(api_key="your_api_key", endpoint="your_endpoint", model_name="gpt-4"),
    ]
    
    # 评估所有模型
    results = []
    for model in models:
        try:
            result = evaluator.evaluate_model(model, test_data)
            results.append(result)
            
            # 打印结果摘要
            print(f"\n{'='*50}")
            print(f"模型: {result.model_name}")
            print(f"总体准确率: {result.accuracy:.4f}")
            print(f"间质瘤 - 精确率: {result.gist_precision:.4f}, 召回率: {result.gist_recall:.4f}, F1: {result.gist_f1:.4f}")
            print(f"非间质瘤 - 精确率: {result.non_gist_precision:.4f}, 召回率: {result.non_gist_recall:.4f}, F1: {result.non_gist_f1:.4f}")
            print(f"执行时间: {result.execution_time:.2f}秒")
            print(f"{'='*50}")
            
        except Exception as e:
            logger.error(f"评估模型 {model.get_model_name()} 失败: {e}")
    
    # 生成模型比较报告
    if len(results) > 1:
        comparison_df = evaluator.compare_models(results)
        print("\n模型性能比较:")
        print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()

