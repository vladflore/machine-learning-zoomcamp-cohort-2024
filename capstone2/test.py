import requests
import argparse

parser = argparse.ArgumentParser(description="Send image URL to server for processing.")
parser.add_argument(
    "--image-url", type=str, required=True, help="The image URL to be processed."
)

args = parser.parse_args()

url = "http://localhost:8080/2015-03-31/functions/function/invocations"
data = {"url": args.image_url}

result = requests.post(url, json=data).json()
print(result)
