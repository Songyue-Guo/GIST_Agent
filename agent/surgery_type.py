from typing import Dict, Any, List
import json
import sys
sys.path.insert(0, '/hpc2hdd/home/sguo349/gsy/hebei4/')
from agent.base_agent import BaseAgent
from model.qwen import ChatQwen
from test_cases import case_early_operable


class SurgeryTypeAgent(BaseAgent):
    """手术方式选择Agent - 根据患者情况推荐合适的手术方式并评分"""
    
    def generate_prompt(self, patient_info: Dict[str, Any]) -> str:
        """
        根据患者信息生成提示词
        
        Args:
            patient_info: 包含患者信息的字典，应至少包含以下字段:
                - basic_info: 基本信息 (年龄、性别等)
                - symptoms: 症状描述
                - examination: 检查结果 (如CT、胃镜等)
                - history: 病史相关信息
                - tumor_info: 肿瘤相关信息 (大小、位置、分期等)
                
        Returns:
            生成的提示词字符串
        """
        # 检索相关知识
        retrieval_question = "胃间质瘤手术方式类型及选择标准"
        knowledge = self.retrieve_knowledge(retrieval_question, {
            "dev_Guidelines_for_GIST": 8,
            "dev_Cases_in_GIST": 6
        })
        
        # 格式化患者信息
        formatted_patient_info = "\n".join([
            f"【基本信息】\n{patient_info.get('basic_info', '无')}",
            f"【症状描述】\n{patient_info.get('symptoms', '无')}",
            f"【检查结果】\n{patient_info.get('examination', '无')}",
            f"【病史信息】\n{patient_info.get('history', '无')}",
            f"【肿瘤信息】\n{patient_info.get('tumor_info', '无')}"
        ])
        
        # 构建提示词
        prompt = f"""
你是一位专业的消化外科医生，专门负责胃肠间质瘤(GIST)的手术方案制定。现在你需要为一位确诊为胃间质瘤且适合手术的患者推荐最合适的手术方式。

请基于以下患者信息和专业知识，评估该患者适合的手术方式并给出推荐分数。

【患者信息】
{formatted_patient_info}

【相关医学知识】
{knowledge}

请你仔细分析以下几个方面:
1. 肿瘤特征: 大小、位置、生长方式(内生型/外生型)、侵袭性
2. 患者一般状况: 年龄、体能状态、基础疾病
3. 手术技术可行性: 考虑肿瘤位置是否适合腹腔镜或开腹
4. 患者期望: 对术后生活质量、恢复速度的需求

【输出格式】
请以JSON格式输出您的评估结果，包括多种可能的手术方案及其评分:
{{
  "recommended_surgeries": [
    {{
      "surgery_type": "手术方式名称",
      "score": 0-100之间的数字,
      "advantages": ["优点1", "优点2"...],
      "disadvantages": ["缺点1", "缺点2"...],
      "rationale": "推荐理由"
    }},
    // 可包含多个手术方案，按评分从高到低排序
  ],
  "reasoning_process": "详细的推理过程",
  "key_considerations": ["考虑因素1", "考虑因素2"...]
}}

请确保至少推荐3种可能的手术方案，并按照评分从高到低排序。每种手术方案都应该详细说明其优缺点和适用条件。
"""
        return prompt
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型返回的响应
        
        Args:
            response: 模型响应文本
            
        Returns:
            包含以下字段的字典:
            - recommended_surgeries: 推荐的手术方式列表，每个元素包含:
              - surgery_type: 手术方式名称
              - score: 评分 (0-100)
              - advantages: 优点列表
              - disadvantages: 缺点列表
              - rationale: 推荐理由
            - reasoning_process: 推理过程
            - key_considerations: 关键考虑因素列表
        """
        try:
            # 尝试解析JSON响应
            result = json.loads(response)
            
            # 确保结果包含所有必需字段
            if 'recommended_surgeries' not in result or not isinstance(result['recommended_surgeries'], list):
                result['recommended_surgeries'] = []
                
            # 处理每个手术方案
            for i, surgery in enumerate(result['recommended_surgeries']):
                # 确保每个手术方案包含所有必需字段
                if 'surgery_type' not in surgery:
                    surgery['surgery_type'] = f"未命名手术方案 {i+1}"
                
                if 'score' not in surgery:
                    surgery['score'] = 0
                    
                if 'advantages' not in surgery or not isinstance(surgery['advantages'], list):
                    surgery['advantages'] = []
                    
                if 'disadvantages' not in surgery or not isinstance(surgery['disadvantages'], list):
                    surgery['disadvantages'] = []
                    
                if 'rationale' not in surgery:
                    surgery['rationale'] = ""
            
            # 按评分从高到低排序
            result['recommended_surgeries'] = sorted(
                result['recommended_surgeries'], 
                key=lambda x: x.get('score', 0), 
                reverse=True
            )
            
            # 确保其他字段存在
            if 'reasoning_process' not in result:
                result['reasoning_process'] = ""
                
            if 'key_considerations' not in result or not isinstance(result['key_considerations'], list):
                result['key_considerations'] = []
                
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试提供一个基本的结构
            return {
                'recommended_surgeries': [
                    {
                        'surgery_type': "无法解析推荐手术方式",
                        'score': 0,
                        'advantages': [],
                        'disadvantages': [],
                        'rationale': "无法从模型响应中解析有效的手术推荐"
                    }
                ],
                'reasoning_process': response,
                'key_considerations': []
            }
if __name__ == "__main__":
    agent = SurgeryTypeAgent(ChatQwen())
    print(agent.make_decision(case_early_operable))