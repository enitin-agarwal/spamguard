# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from flask import Flask, request, jsonify
import re
import requests
import json
from datetime import datetime
import whois
import certifi
import ssl
import socket
import nltk
import openai
import concurrent.futures
import threading
from textblob import TextBlob
#from flask_limiter import Limiter
#from flask_limiter.util import get_remote_address
from werkzeug.exceptions import BadRequest


# Download NLTK resources
nltk.download('punkt')
nltk.download('vader_lexicon')
# Load the dataset (CSV format)
data = pd.read_csv('url_dataset.csv')


# Initialize your LLM API key
llm_api_key = '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'  # Replace with your GPT-3 API key

# Initialize your LLM instance
openai.api_key = llm_api_key



# Initialize an empty feedback dictionary
feedback_data = []
# Create a lock to make feedback_data thread-safe
feedback_data_lock = threading.Lock()


# Preprocess the data
vectorizer = TfidfVectorizer()
X_text = data['url']
X = vectorizer.fit_transform(X_text)
y = data['label']
X = np.hstack((X.toarray(), data[['url_length', 'subdomains', 'special_chars', 'has_keywords','is_https','has_valid_certificate','website_age']].values))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# Hyperparameter tuning for AdaBoost
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0]
}

ab = AdaBoostClassifier(base_estimator=best_rf, random_state=42)
grid_search_ab = GridSearchCV(ab, param_grid, cv=3, n_jobs=-1)
grid_search_ab.fit(X_train, y_train)
best_ab = grid_search_ab.best_estimator_

# Create a Voting Classifier ensemble with Random Forest and AdaBoost
ensemble_classifier = VotingClassifier(estimators=[('rf', best_rf), ('ab', best_ab)], voting='soft')

# Train the ensemble classifier
ensemble_classifier.fit(X_train, y_train)

# Create a Flask application
app = Flask(__name__)

# Add rate limiting to restrict the number of requests from the same IP
'''limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["10 per minute"]
)'''

# Define a function for input validation
def is_valid_url(url):
    # You can add your own validation logic here
    # For example, check if the URL format is valid
    if re.match(r'^https?://', url):
        return True
    return False

# Define a function for URL sanitization
def sanitize_url(url):
    # You can add your own sanitization logic here
    # For example, remove potentially harmful characters
    sanitized_url = re.sub(r'[^a-zA-Z0-9:/._-]', '', url)
    return sanitized_url


# API endpoint for model training
#@app.route('/train_model', methods=['POST'])
def train_model(new_data):
    # Get new data for training (simplified, assumes data is in JSON format)
    #new_data = request.json
    print(new_data)

    # Extract features from new data
    new_urls = [entry["url"] for entry in new_data]
    print(new_urls)
    new_lengths = np.array([len(url) for url in new_urls])
    new_subdomains = np.array([url.count('.') for url in new_urls])
    new_special_chars = np.array([len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url)) for url in new_urls])
    new_has_keywords = np.array([1 if re.search(r'(spam|phish)', url) else 0 for url in new_urls])

    # Perform NLP analysis on URLs
    url_sentiments = [analyze_sentiments(url) for url in new_urls]
    url_subjectivities = [analyze_subjectivity(url) for url in new_urls]

    # Convert sentiment and subjectivity lists to NumPy arrays
    url_sentiments = np.array(url_sentiments)
    url_subjectivities = np.array(url_subjectivities)

    # Preprocess the new data
    new_X_text = np.array(new_urls)
    new_X = vectorizer.transform(new_X_text)
    new_X = np.hstack((new_X.toarray(), new_lengths.reshape(-1, 1), new_subdomains.reshape(-1, 1),
                        new_special_chars.reshape(-1, 1), new_has_keywords.reshape(-1, 1),
                        url_sentiments.reshape(-1, 1), url_subjectivities.reshape(-1, 1)))



    new_y = [entry["label"] for entry in new_data]

    # Retrain the ensemble model with new data
    ensemble_classifier.fit(new_X, new_y)

    #return jsonify({'message': 'Model trained successfully with new data'})

# Function to analyze sentiments in URL text
def analyze_sentiments(url):
    blob = TextBlob(url)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Function to analyze subjectivity in URL text
def analyze_subjectivity(url):
    blob = TextBlob(url)
    subjectivity_score = blob.sentiment.subjectivity
    return subjectivity_score

# Function to count the number of redirections for a URL
def count_redirects(url):
    try:
        response = requests.get(url, allow_redirects=True, timeout=5)
        print(response)
        return len(response.history)
    except Exception as e:
        return 0

# Function to get the creation date of a domain using python-whois library
def domain_creation_date(domain):
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list):
            # If there are multiple creation dates, take the first one
            creation_date = creation_date[0]
            if isinstance(creation_date, datetime):
                return creation_date
        elif isinstance(creation_date, datetime):
            return creation_date
    except Exception as e:
        pass
    return None


# Function to calculate the age of a website (in days)
def website_age_in_days(url):
    try:
        domain = re.findall(r'(?<=://)([^/]+)', url)[0]
        creation_date = domain_creation_date(domain)
        if creation_date:
            current_date = datetime.now()
            age = current_date - creation_date
            print(age)
            return age.days
    except Exception as e:
        pass
    return 0

# Function to use LLM for text analysis
def analyze_text_with_llm(text):
    response = openai.Completion.create(
        engine="text-davinci-002",  # Specify the LLM engine
        prompt=text,  # Provide the URL or text to analyze
        max_tokens=50,  # Adjust the token limit as needed
        temperature=0.7,  # Adjust the temperature for creativity
        n = 1  # Number of completions to generate
    )
    # Extract the LLM-generated text
    llm_text = response.choices[0].text.strip()
    return llm_text


# API endpoint for URL classification
#@app.route('/classify_url', methods=['POST'])
def classify_url(url_to_classify):
    # Get the URL to classify (simplified, assumes data is in JSON format)
    #url_to_classify = request.json['url']

    # Extract features from the URL
    url_length = len(url_to_classify)
    subdomains = url_to_classify.count('.')
    special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', url_to_classify))
    redirects = count_redirects(url_to_classify)
    is_https, has_valid_certificate = check_certificate(url_to_classify)
    website_age = website_age_in_days(url_to_classify)
    has_keywords = 1 if re.search(r'(spam|phish)', url_to_classify) else 0

    # Preprocess the URL
    url_vector = vectorizer.transform([url_to_classify])
    url_vector = np.hstack((url_vector.toarray(), np.array([url_length, subdomains, special_chars, has_keywords, is_https, has_valid_certificate, website_age]).reshape(1, -1)))

    # Classify the URL using the trained model
    prediction = ensemble_classifier.predict(url_vector)
    probability = ensemble_classifier.predict_proba(url_vector)

    # Enhance classification using LLM (GPT-3)
    ''' llm_analysis = analyze_text_with_llm(url_to_classify)
    if llm_analysis:
        print("LLM Analysis:", llm_analysis)
        # Update the prediction based on LLM insights (you can define your logic here)
        if "spam" in llm_analysis.lower():
            prediction[0] = "spammy" '''

    # Save incorrect classifications for feedback
    if prediction[0] != "legitimate":
        feedback_data.append({'url': url_to_classify, 'actual_label': prediction[0]})


    # If it's a false positive or false negative, ask for feedback
    if prediction[0] != "legitimate":
        feedback_data.append({'url': url_to_classify, 'actual_label': prediction[0]})

    #return jsonify({'classification_result': prediction[0]})
    #response = (f'Classification Result: {prediction[0]}', probability[0].tolist())
    #print(response)
    prediction_tuple = (prediction[0])
    probability_tuple = (probability[0].tolist())
    return prediction_tuple,probability_tuple

# API endpoint for users to provide feedback
@app.route('/provide_feedback', methods=['POST'])
def provide_feedback():
    feedback = request.json
    feedback_data.append(feedback)
    return jsonify({'message': 'Feedback received and will be used for model improvement.'})

# Function to check if a URL is a short URL
def is_short_url(url):
    return len(url) <= 20  # You can adjust the threshold as needed

# Function to check certificate validity
def check_certificate(url):
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with context.wrap_socket(socket.socket(), server_hostname=url) as ssock:
            ssock.connect((url, 443))
            cert = ssock.getpeercert()
            if cert and "notAfter" in cert:
                expiration_date = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y GMT")
                return True, expiration_date > datetime.now()
    except Exception as e:
        pass
    return False, False

# API endpoint for model training
@app.route('/train_model_endpoint', methods=['POST'])
def train_model_endpoint():
    # Get new data for training (simplified, assumes data is in JSON format)
    new_data = request.json
    print(new_data)

    # Train the model in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(train_model, new_data)

    return jsonify({'message': 'Model trained successfully with new data'})

# API endpoint for URL classification
@app.route('/classify_url_endpoint', methods=['POST'])
#@limiter.limit("5 per minute")  # Apply rate limiting
def classify_url_endpoint():
    # Get the URL to classify (simplified, assumes data is in JSON format)
    url_to_classify = request.json['url']

    # Validate the URL
    if not is_valid_url(url_to_classify):
        raise BadRequest('Invalid URL format.')

    # Sanitize the URL
    sanitized_url = sanitize_url(url_to_classify)

    # Classify the URL in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = executor.submit(classify_url, url_to_classify)
        #classification_result = result.result()
        return result

    #return jsonify({'classification_result': classification_result})
    #print(classification_result)
    #return classification_result

# Main function to run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
