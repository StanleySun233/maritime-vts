from typing import Dict

import requests
import yaml
from openai import OpenAI


def load_config(config_path: str = "api.yaml") -> Dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise Exception(f"Error loading config file: {str(e)}")


def generate_answer(model: str, question: str, system: str = "You are a helpful assistant.",
                    config_path: str = "api.yaml") -> str:
    config = load_config(config_path)
    if model not in config:
        raise ValueError(f"Model {model} not found in config file")

    model_config = config[model]
    model_type = model_config["type"]

    try:
        if model_type == "openai":
            if "api_key" not in model_config:
                raise ValueError("OpenAI API key is required")

            base_url = model_config.get("base_url", None)
            client = OpenAI(
                api_key=model_config["api_key"],
                base_url=base_url
            )
            model_name = model_config.get("model_name", "gpt-3.5-turbo")

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content

        elif model_type == "ollama":
            # Ollama API调用
            base_url = model_config.get("base_url", "http://localhost:11434")
            model_name = model_config.get("model_name", "llama2")

            # 构建完整的提示词
            full_prompt = f"{system}\n\n{question}"

            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            # response.raise_for_status()
            # print(response.json())
            try:
                return response.json()["response"]
            except:
                print("SQL Error:", response.json())
                return ""

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    except Exception as e:
        raise Exception(f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    test_question = "What is the capital of France?"
    test_system = "You are a helpful assistant that provides accurate and concise answers."

    try:
        # 测试 GPT-4
        print("\n=== Testing GPT-4 ===")
        gpt_response = generate_answer("gpt_4o", test_question, test_system, "../api.yaml")
        print("GPT-4 Response:", gpt_response)

        # # 测试 Qwen
        # print("\n=== Testing Qwen ===")
        # qwen_response = generate_answer("qwen7b", test_question, test_system)
        # print("Qwen Response:", qwen_response)

    except Exception as e:
        print("Test Error:", str(e))
