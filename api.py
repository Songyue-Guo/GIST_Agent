import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from concurrent.futures import ThreadPoolExecutor

from model.qwen import ChatQwen
from agent.surgery_decision import SurgeryDecisionAgent
from agent.surgery_type import SurgeryTypeAgent
from agent.treatment import TreatmentAgent
import argparse


# 初始化FastAPI应用
app = FastAPI(
    title="胃间质瘤决策辅助系统API",
    description="提供胃间质瘤手术决策、手术方式选择和治疗方案推荐的AI辅助决策系统",
    version="1.0.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型和agents
model = ChatQwen()
surgery_decision_agent = SurgeryDecisionAgent(model)
surgery_type_agent = SurgeryTypeAgent(model)
treatment_agent = TreatmentAgent(model)

# 定义API数据模型
class PatientInfo(BaseModel):
    basic_info: str
    symptoms: str
    examination: str
    history: str
    tumor_info: Optional[str] = None
    surgery_info: Optional[str] = None
    pathology: Optional[str] = None

class DecisionResponse(BaseModel):
    surgery_recommended: bool
    confidence_score: float
    reasoning: str
    risk_factors: list
    alternative_recommendations: str

class SurgeryTypeResponse(BaseModel):
    recommended_surgeries: list
    reasoning_process: str
    key_considerations: list

class TreatmentResponse(BaseModel):
    treatment_stage: str
    recommended_drugs: list
    rationale: str
    potential_side_effects: list
    follow_up_plan: str
    alternative_options: str

# 工作线程函数
def run_agent(agent, patient_info):
    return agent.make_decision(patient_info.dict())

# API路由
@app.post("/api/surgery_decision", response_model=DecisionResponse)
async def get_surgery_decision(patient_info: PatientInfo):
    """
    评估患者是否适合进行胃间质瘤手术
    """
    try:
        # 使用线程池执行模型推理以避免阻塞
        with ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, run_agent, surgery_decision_agent, patient_info
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

@app.post("/api/surgery_type", response_model=SurgeryTypeResponse)
async def get_surgery_type(patient_info: PatientInfo):
    """
    为适合手术的胃间质瘤患者推荐手术方式
    """
    try:
        # 使用线程池执行模型推理以避免阻塞
        with ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, run_agent, surgery_type_agent, patient_info
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

@app.post("/api/treatment", response_model=TreatmentResponse)
async def get_treatment(patient_info: PatientInfo):
    """
    为胃间质瘤患者推荐药物治疗方案
    """
    try:
        # 使用线程池执行模型推理以避免阻塞
        with ThreadPoolExecutor() as executor:
            result = await asyncio.get_event_loop().run_in_executor(
                executor, run_agent, treatment_agent, patient_info
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

@app.post("/api/comprehensive_decision")
async def get_comprehensive_decision(patient_info: PatientInfo):
    """
    提供全面的决策支持，包括手术决策、手术方式选择和药物治疗方案
    """
    try:
        results = {}
        
        # 并行执行三个决策任务
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_decision = asyncio.get_event_loop().run_in_executor(
                executor, run_agent, surgery_decision_agent, patient_info
            )
            future_surgery_type = asyncio.get_event_loop().run_in_executor(
                executor, run_agent, surgery_type_agent, patient_info
            )
            future_treatment = asyncio.get_event_loop().run_in_executor(
                executor, run_agent, treatment_agent, patient_info
            )
            
            results['surgery_decision'] = await future_decision
            results['surgery_type'] = await future_surgery_type
            results['treatment'] = await future_treatment
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")

# API启动函数
def start_api(host="0.0.0.0", port=18000):
    """启动API服务器"""
    uvicorn.run("api:app", host=host, port=port, reload=True)

if __name__ == "__main__":

    import argparse

    def parse_arguments():
        parser = argparse.ArgumentParser(description="启动胃间质瘤决策辅助系统API服务")
        parser.add_argument("--port", type=int, default=18000, help="API服务端口，默认为8000")
        return parser.parse_args()

    args = parse_arguments()
    start_api(port=args.port)
