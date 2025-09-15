from openai import AzureOpenAI

class AzureModel:
    def __init__(self, api_key: str = 'dbfa63b54a2744d7aba5c2008b125a86', endpoint: str ='https://mdi-gpt-4o.openai.azure.com/', model_name: str = "gpt-4o-global"):
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