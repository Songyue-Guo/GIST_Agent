
from azure import AzureModel
from openrouter import ClaudeModel, GeminiModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="anthropic/claude-sonnet-4")
    args = parser.parse_args()
    
    model = ClaudeModel(model_name=args.model_name)
    response = model.generate_response("Hello, how are you?")
    print(response)







