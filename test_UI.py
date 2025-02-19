import streamlit as st  # Streamlit for web UI
import json  # For handling JSON responses
import requests  # For sending HTTP requests

# Define the base URL for the API (ensure the Flask API is running on this port)
BASE_URL = "http://127.0.0.1:5001"

# Configure Streamlit app settings
st.set_page_config(page_title="API Tester", layout="centered")
st.title("API Tester - Predict Endpoint")
st.markdown("Use this tool to send test data to the API and view responses.")

# User input section
st.subheader("Input Data")
description = st.text_area(
    "Enter a medical description:", 
    "This is a test description about Dementia."
)

# Display area for API response
st.subheader("Output")
label_output = st.empty()  # Placeholder for displaying prediction

# Button to submit data to the API
if st.button("Submit to API", use_container_width=True):
    if description.strip():  # Ensure input is not empty
        with st.spinner("Sending request..."):  # Show a loading spinner
            test_data = {"description": description}  # Prepare request payload
            
            try:
                # Send POST request to the API
                response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=10)

                # Handle response
                if response.status_code == 200:
                    result = response.json()  # Parse JSON response
                    label = result.get("prediction", "No prediction received.")
                    
                    # Display the prediction result
                    label_output.text_input("Predicted Label:", label, disabled=True)
                else:
                    # Display error message if the request fails
                    st.error(f"API Error: {response.status_code}")
                    st.code(response.text, language="json")
            
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {str(e)}")  # Handle request errors

    else:
        st.warning("Please enter a description before submitting.")  # Warn user if input is empty
