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