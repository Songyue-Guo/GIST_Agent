import openai

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = "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"
class ClaudeModel:
    def __init__(self, model_name: str = "anthropic/claude-sonnet-4",api_key: str = API_KEY):
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=BASE_URL, api_key=api_key)
        
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **kwargs)
        return response.choices[0].message.content

class Qwen3_30b_Model:
    def __init__(self, model_name: str = "qwen/qwen3-30b-a3b:free", api_key: str = API_KEY):
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=BASE_URL, api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **kwargs)
        return response.choices[0].message.content

class GeminiModel:
    def __init__(self, model_name: str = "google/gemini-2.5-flash", api_key: str = API_KEY):
        self.model_name = model_name
        self.client = openai.OpenAI(base_url=BASE_URL, api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], **kwargs)
        return response.choices[0].message.content

if __name__ == "__main__":
    model = Qwen3_30b_Model()
    response = model.generate("Hello, please return a json object with the key 'response' and the value 'Hello, how are you?'",response_format={"type": "json_object"})
    print(response)