import openai

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = "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"
class ClaudeModel:
    def __init__(self, model_name: str = "anthropic/claude-sonnet-4",api_key: str = API_KEY):
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=BASE_URL, api_key=api_key)
        
    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **kwargs)
        return response.choices[0].message.content


class GeminiModel:
    def __init__(self, model_name: str = "google/gemini-2.5-flash", api_key: str = API_KEY):
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=BASE_URL, api_key=api_key)

    def generate_response(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **kwargs)
        return response.choices[0].message.content
    
    