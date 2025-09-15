from openai import OpenAI
import os

class ChatQwen():
    def __init__(self, model_string="Qwen/Qwen3-32B", use_cache=True, **kwargs):
        self.model_string = model_string
        self.use_cache = use_cache
        # if self.use_cache:
        #     cache_path = f"cache_qwen_{model_string}.db"
        #     super().__init__(cache_path=cache_path)
        self.client = OpenAI(api_key='mdi', base_url="https://mdi.hkust-gz.edu.cn/hpc/qwen3/v1")
    
    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.7,
            top_p=0.8,
            presence_penalty=1.5,
            extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    },
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    model = ChatQwen()
    print(model.generate("Give me a short introduction to large language models."))