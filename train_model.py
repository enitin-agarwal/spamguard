import requests
import json

# Define the data as a list of dictionaries
data = [
 {"url": "https://www.newurl1.com", "label": "legitimate", "url_length": 21, "subdomains": 3, "special_chars": 0, "has_keywords": 0},
 {"url": "http://newspammyurl.net", "label": "spammy", "url_length": 18, "subdomains": 2, "special_chars": 1, "has_keywords": 1}
]

#data = {"url": "https://www.bitly.com"}



# Convert data to JSON format
data_json = json.dumps(data)

# Define the API endpoint URL
url = 'http://127.0.0.1:5000/train_model'
#url = 'http://127.0.0.1:5000/classify_url_endpoint'


print(data_json)
# Send a POST request to the API
response = requests.post(url, headers={"Content-Type": "application/json"}, data=data_json)

# Print the response
print(response.json())
