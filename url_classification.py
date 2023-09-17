import requests
import json

data = {"url": "https://www.bitly.com"}



# Convert data to JSON format
data_json = json.dumps(data)

# Define the API endpoint URL
url = 'http://127.0.0.1:5000/classify_url_endpoint'


print(data_json)
# Send a POST request to the API
response = requests.post(url, headers={"Content-Type": "application/json"}, data=data_json)

# Print the response
print(response.json())
