import requests
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from agent.tool.rag_tool import RAGTool
import time

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
            print(f"来源 {i+1} ({source}):\n{content}\n")
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
        start_time = time.time()
        # 1. 生成提示词
        prompt = self.generate_prompt(patient_info)
        
        # 2. 调用模型
        # 检查是否有response_format属性，如果有则传入
        if hasattr(self, 'response_format') and self.response_format:
            if self.response_format == "json":
                response = self.model.generate(prompt, response_format={"type": "json_object"})
            else:
                response = self.model.generate(prompt, response_format=self.response_format)
        else:
            response = self.model.generate(prompt)
            
        #print(response)
        
        # 3. 处理模型响应
        result = self.process_response(response)
        end_time = time.time()
        print(f"Agent执行时间: {end_time - start_time:.2f}秒")
        return result
