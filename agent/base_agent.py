import requests
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

class RAGTool:
    """知识检索工具类，用于从知识库中检索相关信息"""
    
    def __init__(
        self,
        url: str = "https://mdi.hkust-gz.edu.cn/llm/agent/dev_yisrag/api/multi_retrieve",
        token: str = "SK-kLXycej6SgAwJFyQzV2Xjorp4vut24T4"
    ):
        self.url = url
        self.token = token
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}'
        }
    
    def retrieve(self, question: str, sources: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        从知识库检索相关信息
        
        Args:
            question: 检索问题
            sources: 源文档和top_k配置，如 {"dev_Cases_in_GIST": 6, "dev_Guidelines_for_GIST": 6}
            
        Returns:
            检索结果列表
        """
        if sources is None:
            sources = {
                "dev_Cases_in_GIST": 6,
                "dev_Guidelines_for_GIST": 6,
            }
        
        payload = json.dumps({
            "question": question,
            "source_top_k": sources
        })
        
        try:
            response = requests.request("POST", self.url, headers=self.headers, data=payload)
            response.raise_for_status()
            result = json.loads(response.text)
            return result['sources']
        except Exception as e:
            print(f"检索失败: {str(e)}")
            return []

class BaseAgent(ABC):
    """Agent基类，提供通用功能"""
    
    def __init__(self, model):
        """
        初始化Agent
        
        Args:
            model: LLM模型实例，需提供generate_response方法
        """
        self.model = model
        self.rag_tool = RAGTool()
    
    def retrieve_knowledge(self, question: str, sources: Dict[str, int] = None) -> str:
        """
        检索相关知识并格式化为字符串
        
        Args:
            question: 检索问题
            sources: 源文档和top_k配置
            
        Returns:
            格式化的知识字符串
        """
        results = self.rag_tool.retrieve(question, sources)
        if not results:
            return "未找到相关知识"
        
        formatted_results = []
        for i, result in enumerate(results):
            source = result.get("source", "未知来源")
            content = result.get("content", "").strip()
            formatted_results.append(f"来源 {i+1} ({source}):\n{content}\n")
            
        return "\n".join(formatted_results)
    
    @abstractmethod
    def generate_prompt(self, patient_info: Dict[str, Any]) -> str:
        """
        根据患者信息生成提示词
        
        Args:
            patient_info: 患者信息字典
            
        Returns:
            生成的提示词
        """
        pass
    
    @abstractmethod
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        处理模型返回的响应
        
        Args:
            response: 模型响应文本
            
        Returns:
            处理后的结果字典
        """
        pass
    
    def make_decision(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据患者信息做出决策
        
        Args:
            patient_info: 患者信息字典
            
        Returns:
            决策结果字典
        """
        # 1. 生成提示词
        prompt = self.generate_prompt(patient_info)
        
        # 2. 调用模型
        response = self.model.generate(prompt)
        print(response)
        
        # 3. 处理模型响应
        result = self.process_response(response)
        
        return result
