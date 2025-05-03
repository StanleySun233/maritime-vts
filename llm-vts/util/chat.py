import requests
import yaml


def get_answer_by_agent_id(cfg, agent_id, question, stream=False):
    if isinstance(cfg, str):
        with open(cfg, "r", encoding='utf-8') as config_file:
            cfg = yaml.safe_load(config_file)
    base_url = cfg["api"]["address"]
    api_key = cfg["api"]["api_key"]
    url = f"http://{base_url}/api/v1/agents/{agent_id}/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    payload = {
        "question": question,
        "stream": stream
    }

    session_id = requests.post(url, headers=headers, json=payload).json()["data"]["session_id"]
    payload["session_id"] = session_id
    response = requests.post(url, headers=headers, json=payload)
    return response.json()
