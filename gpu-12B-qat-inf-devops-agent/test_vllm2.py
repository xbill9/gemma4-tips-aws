import requests

url = "http://100.30.184.115:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "google/gemma-4-12B-it",
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 10,
    "temperature": 0.0
}

response = requests.post(url, headers=headers, json=data)
print(response.status_code)
print(response.json())
