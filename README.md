# Machine Learning Challenge

This repository contains a dataset of medical trial descriptions. The objective is to build a machine learning model that classifies trials into different disease categories. The trained model is then served via an API built with Flask.

---

## 🏆 **Project Workflow**
1. **Run `Comparison_model.py`**  
   - This is the **first script** to be executed after downloading the repository.  
   - It builds **three basic models**, evaluates them, and selects the best-performing model.  
   - The chosen model is saved for later use in the Flask API.

2. **Run `main.py` (Flask Server)**  
   - Once the model is trained, this script loads the saved model and starts the API on **port 5001**.  
   - The API allows users to submit trial descriptions and receive predictions.

3. **Testing the API**
   - Use `test.py` for **command-line testing**.  
   - Use `test_UI.py` for **UI-based testing with Streamlit**.

---

## 📂 **File Structure**
| File Name           | Description |
|---------------------|-------------|
| `Comparison_model.py` | **First file to run**. Trains and evaluates three models, then saves the best one. |
| `main.py`          | **Flask server** that loads the model and serves predictions. |
| `test.py`          | **Command-line script** for testing the API. |
| `test_UI.py`       | **Streamlit-based UI** for interactive API testing. |
| `requirements.txt` | List of dependencies. Install them using `pip install -r requirements.txt`. |
| `trials.csv`       | Medical trial dataset with descriptions and labels. |
| `README.md`        | This file, providing an overview of the project. |

---

## 📊 **Dataset Overview**
The dataset consists of **medical trial descriptions** with associated disease labels.

| Disease Condition                | Number of Examples |
|----------------------------------|--------------------|
| Dementia                         | 368                |
| ALS                              | 368                |
| Obsessive Compulsive Disorder    | 358                |
| Scoliosis                        | 335                |
| Parkinson’s Disease              | 330                |

Each record consists of:
- **description** → Input text (trial details).
- **label** → Target category (disease condition).
- **nctid** → Unique trial identifier.

---

## 🚀 **How to Set Up and Run the Project**
### Step 1: Install Dependencies
Run the following command to install the required packages:
```sh
pip install -r requirements.txt
```

### Step 2: Train and Save Models
Execute the `Comparison_model.py` script to train models and save the best one:
```sh
python Comparison_model.py
```
This script generates a model file (e.g., `model.pkl`).

### Step 3: Run Flask API
Before running the API, **update the model file path** in `main.py` to match the correct model file saved by `Comparison_model.py`.  
By default, the script may look for `model.pkl`, but if your file has a different name, update it accordingly.

Then, start the API server:
```sh
python main.py
```
After running this, the API will be available at:
```
http://127.0.0.1:5001/
```

### Step 4: Test the API
#### Option 1: Command-line Testing
```sh
python test.py
```

#### Option 2: **UI-based Testing with Streamlit**
Run the Streamlit server:
```sh
streamlit run test_UI.py
```
This starts the **Streamlit interface**, allowing users to test the API via a graphical UI.

---

## 🏗 **Implementation Details**
- The model is **trained using three different approaches** in `Comparison_model.py`.  
- The **best model is saved and used in `main.py`** to make predictions.  
- The **Flask API runs on port 5001**.
- The **Streamlit UI** provides an interactive way to test predictions.
