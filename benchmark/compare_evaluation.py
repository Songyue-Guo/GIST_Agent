#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比评估脚本：比较原版评估vs基于CoT上下文的评估
"""

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
from openai import AzureOpenAI

# 导入自定义模块
from run_evaluation import DiagnosisEvaluator, AzureModel, EvaluationResult
from run_evaluation_with_cot import CoTDiagnosisEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComparisonEvaluator:
    """评估对比器"""
    
    def __init__(self, data_dir: str = "../data/train_data"):
        self.data_dir = data_dir
        self.original_evaluator = DiagnosisEvaluator(data_dir)
        self.cot_evaluator = CoTDiagnosisEvaluator(data_dir)
        
    def run_comparison(self, num_cot_examples: int = 2, max_test_cases: Optional[int] = None) -> Dict[str, Any]:
        """运行对比评估"""
        logger.info("="*60)
        logger.info("开始对比评估：原版 vs CoT增强版")
        logger.info("="*60)
        
        # 初始化模型
        model = AzureModel()
        
        # 加载测试数据
        test_data = self.original_evaluator.load_test_data()
        if not test_data:
            logger.error("没有找到测试数据")
            return {}
        
        # 限制测试案例数量（用于快速测试）
        if max_test_cases:
            test_data = test_data[:max_test_cases]
            logger.info(f"限制测试案例数量为: {max_test_cases}")
        
        logger.info(f"将评估 {len(test_data)} 个测试案例")
        
        # 统计测试数据分布
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
        
        # 检查CoT示例是否可用
        if not self.cot_evaluator.cot_examples:
            logger.error("没有加载到CoT示例，无法进行CoT增强评估")
            return {}
        
        print(f"\nCoT示例分布:")
        total_cot_examples = sum(len(examples) for examples in self.cot_evaluator.cot_examples.values())
        print(f"  总CoT示例数: {total_cot_examples}")
        for category, examples in self.cot_evaluator.cot_examples.items():
            if examples:
                name = disease_names.get(category, category)
                print(f"  {name}: {len(examples)} 个示例")
        
        # 运行原版评估
        logger.info("\n" + "="*60)
        logger.info("1. 运行原版评估（无CoT上下文）")
        logger.info("="*60)
        
        try:
            original_result = self.original_evaluator.evaluate_model(model, test_data)
            logger.info("✅ 原版评估完成")
        except Exception as e:
            logger.error(f"❌ 原版评估失败: {e}")
            return {}
        
        # 运行CoT增强评估
        logger.info("\n" + "="*60)
        logger.info(f"2. 运行CoT增强评估（每案例{num_cot_examples}个示例）")
        logger.info("="*60)
        
        try:
            cot_result = self.cot_evaluator.evaluate_model_with_cot(model, test_data, num_cot_examples)
            logger.info("✅ CoT增强评估完成")
        except Exception as e:
            logger.error(f"❌ CoT增强评估失败: {e}")
            return {}
        
        # 生成对比报告
        comparison_result = self._generate_comparison_report(original_result, cot_result, num_cot_examples)
        
        # 保存对比结果
        self._save_comparison_results(comparison_result)
        
        # 打印对比结果
        self._print_comparison_results(comparison_result)
        
        return comparison_result
    
    def _generate_comparison_report(self, original_result: EvaluationResult, 
                                  cot_result: EvaluationResult, 
                                  num_cot_examples: int) -> Dict[str, Any]:
        """生成对比报告"""
        
        # 计算改进指标
        accuracy_improvement = cot_result.accuracy - original_result.accuracy
        accuracy_improvement_pct = (accuracy_improvement / original_result.accuracy * 100) if original_result.accuracy > 0 else 0
        
        # 计算各类别的改进
        category_improvements = {}
        disease_names = {
            'GIST': '胃肠间质瘤',
            'LEIOMYOMA': '平滑肌瘤',
            'PANCREATIC': '异位胰腺',
            'SCHWANNOMA': '神经鞘瘤',
            'OTHER': '其他'
        }
        
        for category in original_result.per_class_metrics.keys():
            original_metrics = original_result.per_class_metrics[category]
            cot_metrics = cot_result.per_class_metrics[category]
            
            category_improvements[category] = {
                'precision_improvement': cot_metrics['precision'] - original_metrics['precision'],
                'recall_improvement': cot_metrics['recall'] - original_metrics['recall'],
                'f1_improvement': cot_metrics['f1'] - original_metrics['f1'],
                'original_f1': original_metrics['f1'],
                'cot_f1': cot_metrics['f1']
            }
        
        comparison_result = {
            'evaluation_time': datetime.now().isoformat(),
            'test_cases_count': original_result.total_cases,
            'cot_examples_per_case': num_cot_examples,
            'total_cot_examples_used': cot_result.cot_examples_used,
            
            'original_results': {
                'accuracy': original_result.accuracy,
                'correct_predictions': original_result.correct_predictions,
                'execution_time': original_result.execution_time
            },
            
            'cot_results': {
                'accuracy': cot_result.accuracy,
                'correct_predictions': cot_result.correct_predictions,
                'execution_time': cot_result.execution_time
            },
            
            'improvements': {
                'accuracy_improvement': accuracy_improvement,
                'accuracy_improvement_percentage': accuracy_improvement_pct,
                'additional_correct_predictions': cot_result.correct_predictions - original_result.correct_predictions,
                'execution_time_increase': cot_result.execution_time - original_result.execution_time
            },
            
            'category_wise_comparison': {}
        }
        
        # 添加各类别对比
        for category, improvements in category_improvements.items():
            category_name = disease_names.get(category, category)
            comparison_result['category_wise_comparison'][category_name] = {
                '原版F1分数': f"{improvements['original_f1']:.4f}",
                'CoT增强F1分数': f"{improvements['cot_f1']:.4f}",
                'F1提升': f"{improvements['f1_improvement']:.4f}",
                '精确率提升': f"{improvements['precision_improvement']:.4f}",
                '召回率提升': f"{improvements['recall_improvement']:.4f}"
            }
        
        return comparison_result
    
    def _save_comparison_results(self, comparison_result: Dict[str, Any]):
        """保存对比结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保results目录存在
        os.makedirs("results", exist_ok=True)
        
        # 保存详细对比报告
        comparison_file = f"results/evaluation_comparison_{timestamp}.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, ensure_ascii=False, indent=2)
        
        # 生成简化的CSV报告
        csv_data = {
            '评估方法': ['原版评估', 'CoT增强评估'],
            '总体准确率': [
                f"{comparison_result['original_results']['accuracy']:.4f}",
                f"{comparison_result['cot_results']['accuracy']:.4f}"
            ],
            '正确预测数': [
                comparison_result['original_results']['correct_predictions'],
                comparison_result['cot_results']['correct_predictions']
            ],
            '执行时间(秒)': [
                f"{comparison_result['original_results']['execution_time']:.2f}",
                f"{comparison_result['cot_results']['execution_time']:.2f}"
            ]
        }
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = f"results/evaluation_comparison_summary_{timestamp}.csv"
        csv_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        logger.info(f"对比报告已保存到: {comparison_file}")
        logger.info(f"对比摘要已保存到: {csv_file}")
    
    def _print_comparison_results(self, comparison_result: Dict[str, Any]):
        """打印对比结果"""
        print("\n" + "="*60)
        print("📊 评估对比结果")
        print("="*60)
        
        # 基本信息
        print(f"测试案例数: {comparison_result['test_cases_count']}")
        print(f"每案例CoT示例数: {comparison_result['cot_examples_per_case']}")
        print(f"总CoT示例使用数: {comparison_result['total_cot_examples_used']}")
        
        print(f"\n{'方法':<15} {'准确率':<10} {'正确数':<8} {'执行时间(秒)'}")
        print("-" * 50)
        print(f"{'原版评估':<15} {comparison_result['original_results']['accuracy']:<10.4f} "
              f"{comparison_result['original_results']['correct_predictions']:<8} "
              f"{comparison_result['original_results']['execution_time']:.2f}")
        print(f"{'CoT增强评估':<15} {comparison_result['cot_results']['accuracy']:<10.4f} "
              f"{comparison_result['cot_results']['correct_predictions']:<8} "
              f"{comparison_result['cot_results']['execution_time']:.2f}")
        
        # 改进指标
        improvements = comparison_result['improvements']
        print(f"\n🚀 改进效果:")
        print(f"  准确率提升: {improvements['accuracy_improvement']:.4f} "
              f"({improvements['accuracy_improvement_percentage']:+.2f}%)")
        print(f"  额外正确预测: {improvements['additional_correct_predictions']} 个")
        print(f"  执行时间增加: {improvements['execution_time_increase']:.2f} 秒")
        
        # 各类别对比
        print(f"\n📈 各病种F1分数对比:")
        print("-" * 60)
        print(f"{'病种':<12} {'原版F1':<10} {'CoT F1':<10} {'提升'}")
        print("-" * 60)
        
        for category, metrics in comparison_result['category_wise_comparison'].items():
            print(f"{category:<12} {metrics['原版F1分数']:<10} {metrics['CoT增强F1分数']:<10} {metrics['F1提升']}")
        
        # 结论
        if improvements['accuracy_improvement'] > 0:
            print(f"\n✅ 结论: CoT上下文显著提升了模型诊断准确率!")
        elif improvements['accuracy_improvement'] == 0:
            print(f"\n➖ 结论: CoT上下文对准确率无明显影响")
        else:
            print(f"\n❌ 结论: CoT上下文略微降低了准确率，可能需要调整示例数量或质量")

def main():
    """主函数"""
    print("="*60)
    print("🔍 大模型诊断准确率对比评估系统")
    print("比较原版评估 vs CoT增强评估")
    print("="*60)
    
    # 创建对比评估器
    comparator = ComparisonEvaluator()
    
    # 运行对比评估
    # 可以调整以下参数：
    # - num_cot_examples: 每个测试案例使用的CoT示例数量
    # - max_test_cases: 限制测试案例数量（用于快速测试，None表示使用全部）
    try:
        comparison_result = comparator.run_comparison(
            num_cot_examples=2,      # 每案例使用2个CoT示例
            max_test_cases=None      # 使用所有测试案例，可以设置为小数字进行快速测试
        )
        
        if comparison_result:
            print(f"\n🎉 对比评估完成！")
            print(f"详细结果已保存到 results/ 目录")
        
    except Exception as e:
        logger.error(f"对比评估失败: {e}")

if __name__ == "__main__":
    main() 