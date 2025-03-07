import pandas as pd
import requests


def fetch_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


def process_data(data):
    df = pd.DataFrame(data)
    return df.describe()


if __name__ == "__main__":
    url = "https://jsonplaceholder.typicode.com/posts"
    data = fetch_data(url)
    summary = process_data(data)
    print(summary)
    print("hello!!")
