# Import necessary libraries
import pandas as pd
import numpy as np
import joblib # For loading scikit-learn models and scaler
import tensorflow as tf # For loading the Keras model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
print("--- Model Evaluation Script ---")

try:
    # --- 1. Load and Prepare the Test Data ---
    print("\n[1/4] Loading and preparing test data...")

    # Load the full feature set
    features_df = pd.read_csv("features.csv")

    # Separate features (X) and target (y)
    X = features_df.drop('genre_label', axis=1)
    y = features_df['genre_label']

    # CRUCIAL: Split the data using the *exact same* parameters as in training
    # This ensures we get the identical test set that the models have never seen.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("Test data loaded and split successfully.")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # --- 2. Load the scikit-learn Models and the Scaler ---
    print("\n[2/4] Loading scikit-learn models and scaler...")

    # Load the scaler object
    scaler = joblib.load('scaler.joblib')

    # Load the trained models
    log_reg_model = joblib.load('logistic_regression_model.joblib')
    svm_model = joblib.load('svm_model.joblib')
    rf_model = joblib.load('random_forest_model.joblib')

    print("Scikit-learn assets loaded successfully.")
    print(f"Scaler: {type(scaler)}")
    print(f"Logistic Regression Model: {type(log_reg_model)}")
    print(f"SVM Model: {type(svm_model)}")
    print(f"Random Forest Model: {type(rf_model)}")

    # --- 3. Load the Keras CNN Model ---
    print("\n[3/4] Loading Keras CNN model...")

    # Use TensorFlow/Keras to load the HDF5 file
    cnn_model = tf.keras.models.load_model('music_genre_cnn.h5')

    print("Keras CNN model loaded successfully.")
    print(f"CNN Model: {type(cnn_model)}")
    # Optional: Display the CNN model's architecture
    # cnn_model.summary()

    # --- 4. Prepare Test Data for Different Model Types ---
    print("\n[4/4] Preparing test data for model predictions...")

    # For scikit-learn models, we only need to scale the data.
    X_test_scaled = scaler.transform(X_test)
    print(f"Shape of X_test_scaled (for scikit-learn): {X_test_scaled.shape}")

    # For the CNN model, we need to scale AND reshape the data to 3D.
    X_test_cnn = np.expand_dims(X_test_scaled, axis=-1)
    print(f"Shape of X_test_cnn (for Keras): {X_test_cnn.shape}")
    
    print("\nAll models and data are loaded and ready for evaluation!")

    # --- 5. Generate Predictions for Each Model ---
    print("\n[5/5] Generating predictions on the test set...")

    # --- a) Scikit-learn Models ---
    # The .predict() method directly returns the predicted class label (0-9).
    y_pred_log_reg = log_reg_model.predict(X_test_scaled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    print("Predictions generated for scikit-learn models.")

    # --- b) Keras CNN Model ---
    # The .predict() method returns a 2D array of class probabilities.
    y_pred_cnn_probs = cnn_model.predict(X_test_cnn)
    # We use np.argmax() to find the index of the class with the highest probability.
    # The 'axis=1' argument is crucial: it tells argmax to find the maximum value
    # for each sample (row) across all the class probabilities (columns).
    y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)

    print("Predictions generated for Keras CNN model.")

    # --- Verification Step ---
    # Let's check the shapes of our prediction arrays. They should all have
    # a length equal to the number of samples in our test set (2500).
    print("\n--- Verifying Prediction Shapes ---")
    print(f"Logistic Regression Predictions Shape: {y_pred_log_reg.shape}")
    print(f"SVM Predictions Shape: {y_pred_svm.shape}")
    print(f"Random Forest Predictions Shape: {y_pred_rf.shape}")
    print(f"CNN Predictions Shape: {y_pred_cnn.shape}")

    print("\nAll predictions have been generated successfully!")

    # --- 6. Generate Detailed Classification Reports ---
    
    # Define the genre names for a more readable report.
    # The order must match the integer labels (0=blues, 1=classical, etc.)
    genre_names = [
        'blues', 'classical', 'country', 'disco', 'hiphop', 
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]

    print("\n" + "="*60)
    print("      Classification Report: Logistic Regression")
    print("="*60)
    # The classification_report function takes the true labels and predicted labels.
    # The `target_names` argument makes the output easy to read.
    print(classification_report(y_test, y_pred_log_reg, target_names=genre_names))

    print("\n" + "="*60)
    print("      Classification Report: Support Vector Machine (SVM)")
    print("="*60)
    print(classification_report(y_test, y_pred_svm, target_names=genre_names))

    print("\n" + "="*60)
    print("      Classification Report: Random Forest")
    print("="*60)
    print(classification_report(y_test, y_pred_rf, target_names=genre_names))

    print("\n" + "="*60)
    print("      Classification Report: Convolutional Neural Network (CNN)")
    print("="*60)
    print(classification_report(y_test, y_pred_cnn, target_names=genre_names))

    # --- 7. Compute the Confusion Matrix for Each Model ---
    
    print("\n" + "="*60)
    print("           Computing Confusion Matrices")
    print("="*60)

    # For each model, we compute the confusion matrix. This function returns
    # a 2D NumPy array where rows are the true labels and columns are the
    # predicted labels.
    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    print("\n--- Logistic Regression Confusion Matrix (raw) ---")
    print(cm_log_reg)
    print(f"Shape: {cm_log_reg.shape}") # Should be (10, 10)

    print("\n--- SVM Confusion Matrix (raw) ---")
    print(cm_svm)

    print("\n--- Random Forest Confusion Matrix (raw) ---")
    print(cm_rf)

    print("\n--- CNN Confusion Matrix (raw) ---")
    print(cm_cnn)

    print("\nConfusion matrices computed successfully.")

 # --- 8. Visualize the Confusion Matrices as Heatmaps ---
    
    # We create a helper function for plotting to avoid repetitive code.
    def plot_confusion_matrix(cm, labels, title, ax):
        """
        Plots a confusion matrix as a heatmap using Seaborn.
        
        Args:
            cm (np.array): The confusion matrix to plot.
            labels (list): The list of class names for the axes.
            title (str): The title for the plot.
            ax (matplotlib.axis): The subplot axis to plot on.
        """
        # Create a heatmap using seaborn.
        sns.heatmap(
            cm,                  # The confusion matrix data
            annot=True,          # Annotate each cell with its value
            fmt='d',             # Format the annotation as an integer
            cmap='Blues',        # Use the 'Blues' color map
            xticklabels=labels,  # Set the x-axis labels
            yticklabels=labels,  # Set the y-axis labels
            ax=ax                # Plot on the provided subplot axis
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

    # Create a 2x2 subplot figure to display all four matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrices for All Models', fontsize=20)

    # Plot each confusion matrix on its respective subplot
    plot_confusion_matrix(cm_log_reg, genre_names, 'Logistic Regression', axes[0, 0])
    plot_confusion_matrix(cm_svm, genre_names, 'Support Vector Machine', axes[0, 1])
    plot_confusion_matrix(cm_rf, genre_names, 'Random Forest', axes[1, 0])
    plot_confusion_matrix(cm_cnn, genre_names, 'Convolutional Neural Network', axes[1, 1])

    # Adjust the layout to prevent titles and labels from overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect is used to make space for suptitle

    # Display the final figure
    plt.show()

except FileNotFoundError as e:
    print(f"\nERROR: A required file was not found: {e.filename}")
    print("Please ensure all model files ('scaler.joblib', '*.joblib', 'music_genre_cnn.h5') and 'features.csv' are in the correct directory.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")




