import requests
import json

def main():
    url = "http://52.91.5.186:8080/health"
    print(f"Querying {url}...")
    try:
        r = requests.get(url, timeout=5)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"Error: {e}")

    url_models = "http://52.91.5.186:8080/v1/models"
    print(f"Querying {url_models}...")
    try:
        r = requests.get(url_models, timeout=5)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
