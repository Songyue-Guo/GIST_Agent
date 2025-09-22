from typing import Dict, Any, List
import json
import sys
sys.path.insert(0, '/hpc2hdd/home/sguo349/gsy/hebei4/')
from agent.base_agent import BaseAgent
from model.qwen import ChatQwen
from test_cases import case_early_operable, case_real_early_simple,case_neoadjuvant, case_advanced_unresectable, case_adjuvant, case_recurrent
from model.openrouter import ClaudeModel, Qwen3_30b_Model

class TreatmentAgent(BaseAgent):
    """治疗方案Agent - 推荐胃间质瘤的药物治疗方案"""
    
    def __init__(self, model, response_format="json"):
        super().__init__(model)
        self.model = model
        self.response_format = response_format
    
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
                - surgery_info: 手术相关信息 (是否进行了手术，手术方式等)
                - pathology: 病理检查结果 (免疫组化等)
                
        Returns:
            生成的提示词字符串
        """
        # 检索相关知识 TODO 根据病人信息检索相似病例和对应用药方案
        retrieval_question = "胃间质瘤药物治疗方案和用药指导"
        # knowledge = self.retrieve_knowledge(retrieval_question, {
        #     "dev_Guidelines_for_GIST": 10,
        #     "dev_Cases_in_GIST": 4
        # })
        knowledge = ""
        # 格式化患者信息
        formatted_patient_info = "\n".join([
            f"【基本信息】\n{patient_info.get('basic_info', '无')}",
            f"【症状描述】\n{patient_info.get('symptoms', '无')}",
            f"【检查结果】\n{patient_info.get('examination', '无')}",
            f"【病史信息】\n{patient_info.get('history', '无')}",
            f"【肿瘤信息】\n{patient_info.get('tumor_info', '无')}",
            f"【手术信息】\n{patient_info.get('surgery_info', '无')}",
            f"【病理检查】\n{patient_info.get('pathology', '无')}"
        ])
        
        # 构建提示词
        prompt = f"""
你是一位专业的消化肿瘤科医生，专门负责胃肠间质瘤(GIST)的药物治疗方案制定。现在你需要为一位胃间质瘤患者推荐最合适的药物治疗方案。

请基于以下患者信息和专业知识，评估该患者适合的药物治疗方案。

【患者信息】
{formatted_patient_info}

【相关医学知识】
{knowledge}

请你仔细分析以下几个方面:
1. 肿瘤病理特征: 风险分级、免疫组化结果(尤其是KIT、PDGFRA突变状态)
2. 患者治疗状态: 是术前新辅助治疗、术后辅助治疗还是针对晚期/转移性疾病的治疗
3. 患者一般状况: 年龄、体能状态、肝肾功能
4. 耐药情况: 如果患者曾接受过靶向药物治疗，是否出现耐药

【输出格式】
请以JSON格式输出您的治疗方案推荐:
{{
  "treatment_stage": "术前新辅助/术后辅助/晚期或转移性疾病治疗/密切观察",
  "recommended_drugs": [
    {{
      "drug_name": "药物名称",
      "dosage": "推荐剂量",
      "administration": "用药方式",
      "duration": "推荐疗程",
      "monitoring": "需要监测的指标"
    }},
    // 可包含多个药物
  ],
  "rationale": "治疗方案推荐理由",
  "potential_side_effects": ["副作用1", "副作用2"...],
  "follow_up_plan": "随访计划",
  "alternative_options": "替代治疗方案"
}}

请确保您的治疗方案推荐符合最新的胃肠间质瘤治疗指南，并根据患者的个体特征进行个体化调整。
"""
        return prompt
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型返回的响应
        
        Args:
            response: 模型响应文本
            
        Returns:
            包含以下字段的字典:
            - treatment_stage: 治疗阶段
            - recommended_drugs: 推荐的药物列表，每个元素包含:
              - drug_name: 药物名称
              - dosage: 推荐剂量
              - administration: 用药方式
              - duration: 推荐疗程
              - monitoring: 需要监测的指标
            - rationale: 治疗方案推荐理由
            - potential_side_effects: 潜在副作用列表
            - follow_up_plan: 随访计划
            - alternative_options: 替代治疗方案
        """
        try:
            # 尝试解析JSON响应
            result = json.loads(response)
            
            # 确保结果包含所有必需字段
            if 'treatment_stage' not in result:
                result['treatment_stage'] = "未指定"
                
            # 处理推荐药物
            if 'recommended_drugs' not in result or not isinstance(result['recommended_drugs'], list):
                result['recommended_drugs'] = []
                
            for i, drug in enumerate(result.get('recommended_drugs', [])):
                # 确保每个药物推荐包含所有必需字段
                if 'drug_name' not in drug:
                    drug['drug_name'] = f"未命名药物 {i+1}"
                    
                for field in ['dosage', 'administration', 'duration', 'monitoring']:
                    if field not in drug:
                        drug[field] = "未指定"
            
            # 确保其他字段存在
            if 'rationale' not in result:
                result['rationale'] = ""
                
            if 'potential_side_effects' not in result or not isinstance(result['potential_side_effects'], list):
                result['potential_side_effects'] = []
                
            if 'follow_up_plan' not in result:
                result['follow_up_plan'] = ""
                
            if 'alternative_options' not in result:
                result['alternative_options'] = ""
                
            return result
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试提供一个基本的结构
            return {
                'treatment_stage': "未能确定",
                'recommended_drugs': [],
                'rationale': "无法从模型响应中解析有效的治疗方案",
                'potential_side_effects': [],
                'follow_up_plan': "",
                'alternative_options': ""
            }

if __name__ == "__main__":
    
    agent_with_json = TreatmentAgent(Qwen3_30b_Model(), "json")
    print("使用JSON格式响应：")
    print(agent_with_json.make_decision(case_real_early_simple))
    
    # 不使用response_format参数
    agent_without_format = TreatmentAgent(Qwen3_30b_Model())
    print("\n不使用response_format参数：")
    print(agent_without_format.make_decision(case_real_early_simple))