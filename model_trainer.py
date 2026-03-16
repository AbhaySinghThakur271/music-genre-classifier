# Import the pandas library for data manipulation
import pandas as pd
# Import the function for splitting data into train and test sets
from sklearn.model_selection import train_test_split
# Import the StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Import our models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# Import the accuracy metric for evaluation
from sklearn.metrics import accuracy_score
# Import joblib for saving our models and scaler
import joblib
# Import LabelEncoder and numpy for previous steps
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Define the path to our dataset
CSV_PATH = "features.csv"

# Load the dataset from the CSV file into a pandas DataFrame
try:
    features_df = pd.read_csv(CSV_PATH)
    print("Dataset loaded successfully!")

    # Separate the features (X) from the target label (y)
    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']

    # --- Label Encoding and Splitting Data (code omitted for brevity) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Scaling Features (code omitted for brevity) ---
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nFeatures have been scaled.")

    # --- Training Models (code omitted for brevity) ---
    print("\n--- Training Logistic Regression Model ---")
    log_reg = LogisticRegression(max_iter=1000).fit(X_train_scaled, y_train)
    print("Logistic Regression model trained successfully!")
    
    print("\n--- Training Support Vector Machine (SVM) Model ---")
    svm_model = SVC(probability=True, random_state=42).fit(X_train_scaled, y_train)
    print("Support Vector Machine model trained successfully!")

    print("\n--- Training Random Forest Classifier Model ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
    print("Random Forest Classifier model trained successfully!")

    # --- Evaluating Models (from previous task, code omitted for brevity) ---
    print("\n--- Evaluating Models on the Test Set ---")
    print(f"Logistic Regression Accuracy: {log_reg.score(X_test_scaled, y_test) * 100:.2f}%")
    print(f"Support Vector Machine Accuracy: {svm_model.score(X_test_scaled, y_test) * 100:.2f}%")
    print(f"Random Forest Classifier Accuracy: {rf_model.score(X_test_scaled, y_test) * 100:.2f}%")

    # --- THIS IS THE NEW CODE BLOCK TO ADD ---

    # Save the trained models and the scaler for later use
    print("\n--- Saving Models and Scaler to Disk ---")

    # The first argument is the Python object to save.
    # The second argument is the desired filename.
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(log_reg, 'logistic_regression_model.joblib')
    joblib.dump(svm_model, 'svm_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')

    print("Scaler and models have been successfully saved to disk.")
    print("The following files have been created in your project directory:")
    print("- scaler.joblib")
    print("- logistic_regression_model.joblib")
    print("- svm_model.joblib")
    print("- random_forest_model.joblib")

    # --- END OF THE NEW CODE BLOCK ---

except FileNotFoundError:
    print(f"Error: The file at '{CSV_PATH}' was not found.")
    print("Please ensure 'features.csv' is in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")