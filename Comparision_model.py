# Import necessary libraries
import pandas as pd  # For data manipulation
import re  # For text cleaning using regular expressions
import pickle  # For saving models and vectorizers
import time  # For measuring training time
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer  # For text feature extraction
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.naive_bayes import MultinomialNB  # Naïve Bayes model
from sklearn.svm import SVC  # Support Vector Machine (SVM) model
from sklearn.metrics import accuracy_score, classification_report  # For model evaluation

# Load dataset from CSV file
df = pd.read_csv("data/trials.csv")

# Function to clean text data
def clean_text(text):
    """
    Cleans input text by converting to lowercase and removing non-alphanumeric characters.
    """
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # Remove special characters
    return text

# Apply text cleaning function to the 'description' column
df["description"] = df["description"].astype(str).apply(clean_text)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df["description"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Initialize TF-IDF Vectorizer with stop words removal and feature limitation
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

# Transform training and testing text data into TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save the fitted TF-IDF vectorizer for future use
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Define machine learning models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),  # Logistic Regression Model
    "Naïve Bayes": MultinomialNB(),  # Multinomial Naïve Bayes Model
    "Support Vector Machine": SVC(kernel="linear")  # Support Vector Machine with a linear kernel
}

# Dictionary to store model evaluation results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    start_time = time.time()  # Start timing model training
    
    model.fit(X_train_tfidf, y_train)  # Train the model
    y_pred = model.predict(X_test_tfidf)  # Predict labels for test set
    
    end_time = time.time()  # End timing model training
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    training_time = end_time - start_time  # Compute training duration
    
    # Store model performance results
    results[model_name] = {
        "Accuracy": accuracy,
        "Precision": report["weighted avg"]["precision"],
        "Recall": report["weighted avg"]["recall"],
        "F1-Score": report["weighted avg"]["f1-score"],
        "Training Time": training_time
    }
    
    # Save trained model for future use
    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Convert results dictionary into a DataFrame for better readability
results_df = pd.DataFrame(results).T

# Print model performance results
print(results_df)