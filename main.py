# Import necessary libraries
import pickle  # For saving and loading trained models
import logging  # For logging error and info messages
import re  # For text preprocessing
import numpy as np  # For numerical operations (though not used in the script)
import pandas as pd  # For handling data (not used directly in this script)
from flask import Flask, request, jsonify  # Flask for API development
from sklearn.model_selection import train_test_split  # For splitting dataset (not used in this script)
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF feature extraction
from sklearn.linear_model import LogisticRegression  # Machine learning model
from typing import Literal  # For defining possible condition labels

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# Define file paths for the model and vectorizer
model_path = "Logistic Regression.pkl"  # Path to the trained model
vectorizer_path = "vectorizer.pkl"  # Path to the saved TF-IDF vectorizer

# Define possible condition labels
LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]

# Load the trained model and vectorizer
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)  # Load trained model
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)  # Load TF-IDF vectorizer
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model/vectorizer: {str(e)}")
    model, vectorizer = None, None  # Set to None if loading fails

# Initialize Flask app
app = Flask(__name__)

# Function to clean input text (for preprocessing before prediction)
def clean_text(text):
    """
    Cleans the input text by:
    - Converting text to lowercase
    - Removing non-alphanumeric characters (except spaces)
    - Replacing multiple spaces with a single space
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Function to predict the medical condition based on input text
def predict_condition(description: str) -> LABELS:
    """
    Predicts a medical condition based on the input text.
    - Ensures the model and vectorizer are loaded before making predictions.
    - Cleans the input text using the `clean_text` function.
    - Transforms the cleaned text using the trained TF-IDF vectorizer.
    - Returns the predicted condition label.
    """
    if model is None or vectorizer is None:
        raise ValueError("Model or vectorizer not available.")
    
    processed_text = clean_text(description)  # Preprocess input text
    processed_text_tfidf = vectorizer.transform([processed_text])  # Transform using TF-IDF
    return model.predict(processed_text_tfidf)[0]  # Return predicted condition

# Define a simple route to check if the server is running
@app.route("/")
def hello_world():
    """
    Basic route to verify that the API is running.
    """
    return "Hello, World!"

# Define an API endpoint for making predictions
@app.route("/predict", methods=["POST"])
def identify_condition():
    """
    API endpoint to predict a condition based on the input description.
    - Expects a JSON request with a "description" field.
    - Returns a JSON response containing the predicted condition or an error message.
    """
    try:
        data = request.get_json(force=True)  # Parse incoming JSON request
        
        # Validate that 'description' field is provided
        if "description" not in data:
            return jsonify({"error": "Invalid input: 'description' field is required"}), 400
        
        # Predict the condition based on the description
        prediction = predict_condition(data["description"])
        return jsonify({"prediction": str(prediction)})  # Return prediction as JSON response
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500  # Return a generic error response

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
