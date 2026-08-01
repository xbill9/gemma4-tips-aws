import requests

url = "http://54.88.214.232:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "google/gemma-4-12B-it",
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 100,
    "temperature": 0.0,
    "logprobs": True,
    "top_logprobs": 2
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.json())
