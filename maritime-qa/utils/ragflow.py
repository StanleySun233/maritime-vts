import requests
import yaml

def get_answer_by_agent_id(url, api_key, agent_id, question, stream=False):
    request_url = f"{url}/api/v1/agents/{agent_id}/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    payload = {
        "question": question,
        "stream": stream
    }

    session_id = requests.post(request_url, headers=headers, json=payload).json()["data"]["session_id"]
    payload["session_id"] = session_id
    response = requests.post(request_url, headers=headers, json=payload)
    return response.json()["data"]["answer"]

if __name__ == "__main__":
    api_key = "ragflow-dhYmFmYmRjZjUyNDExZWZiOTUzMDI0Mm"
    agent_id = "c1f81c4803b911f088a10242ac160007"
    base_url = "http://172.20.116.91:1080"
    question = "Which vessels lost communication with the shore base for over 5 minutes in the past 15-min, and what are their name and last known positions?"
    print(get_answer_by_agent_id(base_url, api_key, agent_id, question))