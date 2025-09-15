from typing import Dict, Any
import json
import sys
sys.path.insert(0, '/hpc2hdd/home/sguo349/gsy/hebei4/')
from agent.base_agent import BaseAgent
from model.qwen import ChatQwen
from test_cases import case_early_operable, case_neoadjuvant, case_advanced_unresectable, case_adjuvant, case_recurrent
class SurgeryDecisionAgent(BaseAgent):
    """手术决策评估Agent - 判断患者是否符合手术条件"""
    
    def generate_prompt(self, patient_info: Dict[str, Any]) -> str:
        """
        根据患者信息生成提示词
        
        Args:
            patient_info: 包含患者信息的字典，应至少包含以下字段:
                - basic_info: 基本信息 (年龄、性别等)
                - symptoms: 症状描述
                - examination: 检查结果 (如CT、胃镜等)
                - history: 病史相关信息
                
        Returns:
            生成的提示词字符串
        """
        # 检索相关知识
        retrieval_question = "胃间质瘤手术适应症和禁忌症"
        knowledge = self.retrieve_knowledge(retrieval_question, {
            "dev_Guidelines_for_GIST": 8,
            "dev_Cases_in_GIST": 4
        })
        
        # 格式化患者信息
        formatted_patient_info = "\n".join([
            f"【基本信息】\n{patient_info.get('basic_info', '无')}",
            f"【症状描述】\n{patient_info.get('symptoms', '无')}",
            f"【检查结果】\n{patient_info.get('examination', '无')}",
            f"【病史信息】\n{patient_info.get('history', '无')}"
        ])
        
        # 构建提示词
        prompt = f"""
你是一位专业的消化外科医生，专门负责胃肠间质瘤(GIST)的手术评估。现在你需要判断一位胃间质瘤患者是否符合手术条件。

请基于以下患者信息和专业知识，评估该患者是否适合进行手术。

【患者信息】
{formatted_patient_info}

【相关医学知识】
{knowledge}

请你仔细分析以下几个方面:
1. 肿瘤特征: 大小、位置、侵袭性表现
2. 患者一般状况: 年龄、体能状态、基础疾病
3. 手术风险评估: 有无手术禁忌症、麻醉风险
4. 预期获益分析: 手术对患者生存质量和预后的影响

【输出格式】
请以JSON格式输出您的评估结果，包括:
{{"surgery_recommended": true或false, // 是否推荐手术
  "confidence_score": 0-100之间的数字, // 推荐的置信度分数
  "reasoning": "详细的推理过程", // 决策理由
  "risk_factors": ["风险因素1", "风险因素2"...], // 识别出的主要风险因素
  "alternative_recommendations": "如不推荐手术，建议的替代治疗方案"
}}

请注意，你的决策必须是负责任的医疗建议，需要考虑患者的整体利益和循证医学证据。
"""
        return prompt
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型返回的响应
        
        Args:
            response: 模型响应文本
            
        Returns:
            包含以下字段的字典:
            - surgery_recommended: 是否推荐手术 (bool)
            - confidence_score: 推荐的置信度分数 (0-100)
            - reasoning: 决策理由
            - risk_factors: 风险因素列表
            - alternative_recommendations: 替代建议
        """
        try:
            # 尝试解析JSON响应
            result = json.loads(response)
            
            # 确保结果包含所有必需字段
            required_fields = ['surgery_recommended', 'confidence_score', 'reasoning']
            for field in required_fields:
                if field not in result:
                    result[field] = None
            
            # 确保risk_factors是列表
            if 'risk_factors' not in result or not isinstance(result['risk_factors'], list):
                result['risk_factors'] = []
                
            # 确保alternative_recommendations存在
            if 'alternative_recommendations' not in result:
                result['alternative_recommendations'] = ""
            
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试从文本中提取关键信息
            surgery_recommended = 'recommend' in response.lower() and 'not recommend' not in response.lower()
            
            return {
                'surgery_recommended': surgery_recommended,
                'confidence_score': 50,  # 默认中等置信度
                'reasoning': response,
                'risk_factors': [],
                'alternative_recommendations': ""
            }

if __name__ == "__main__":
    agent = SurgeryDecisionAgent(ChatQwen())
    print(agent.make_decision(case_early_operable))
