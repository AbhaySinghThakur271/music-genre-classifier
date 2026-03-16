# app.py
# This script serves as the main entry point for our Streamlit web application.
# Its purpose is to create a user interface where users can upload an audio
# file and see the predicted music genre from our trained CNN model.

# --- Import Necessary Libraries ---

# For building the web app user interface
import streamlit as st
# For loading and using our trained Keras model
import tensorflow as tf
# For numerical operations and data manipulation
import numpy as np
# For audio processing (loading files, extracting features)
import librosa
# For loading the scaler object
import joblib

# --- Helper Functions ---

# We use this decorator to cache the model and scaler loading process.
# This means the model and scaler are loaded only once when the app starts,
# significantly speeding up subsequent predictions.
@st.cache_data
def load_model_and_scaler():
    """
    Loads the pre-trained Keras CNN model and the StandardScaler object.
    The @st.cache_data decorator ensures this function is run only once.
    """
    try:
        # Load the Keras model. compile=False is a performance optimization for inference.
        model = tf.keras.models.load_model('music_genre_cnn.h5', compile=False)
        # Load the scaler object from the file.
        scaler = joblib.load('scaler.joblib')
        # Define the genre mapping based on the training labels.
        genre_mapping = {
            0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
            5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
        }
        return model, scaler, genre_mapping
    except FileNotFoundError as e:
        st.error(f"Error loading model or scaler file: {e}. Please ensure the files are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading model/scaler: {e}")
        st.stop()

def extract_features(audio_file, sample_rate=22050, n_mfcc=13, n_chroma=12):
    """
    Extracts a feature vector from a single audio file.
    """
    try:
        y, sr = librosa.load(audio_file, sr=sample_rate, duration=30)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spec_roll_mean = np.mean(spec_roll)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        features = np.concatenate([
            mfccs_mean, chroma_mean, np.array([spec_cent_mean]),
            np.array([spec_roll_mean]), np.array([zcr_mean])
        ])
        return features
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def predict_genre(audio_file):
    """
    The main prediction pipeline. It takes an audio file, processes it,
    and returns the predicted genre.
    """
    model, scaler, genre_mapping = load_model_and_scaler()
    features = extract_features(audio_file)
    if features is None:
        return "Error: Could not process audio file. Please try a different file."
    
    try:
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
    except Exception as e:
        st.error(f"Error during feature scaling: {e}")
        return "Error: Feature scaling failed."
        
    features_cnn = np.expand_dims(features_scaled, axis=-1)
    
    try:
        prediction_probs = model.predict(features_cnn)
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return "Error: Model prediction failed."
        
    predicted_index = np.argmax(prediction_probs)
    predicted_genre = genre_mapping.get(predicted_index, "Unknown Genre")
    return predicted_genre

# --- Main Application Logic ---

def main():
    """
    The primary function that builds and runs our Streamlit application.
    """
    st.title("🎵 Music Genre Classification App")
    st.write(
        "Welcome! This application uses a Convolutional Neural Network (CNN) to "
        "predict the genre of a music track."
    )
    st.write(
        "**Instructions:** Please upload a short audio file in `.wav` format to get started."
    )

    uploaded_file = st.file_uploader(
        "Drag and drop your audio file here",
        type=['wav']
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        with st.spinner("Classifying your track... 🎶"):
            predicted_genre = predict_genre(uploaded_file)
        
        st.subheader("Prediction Result")
        st.markdown(f"## **{predicted_genre.capitalize()}**")

if __name__ == '__main__':
    main()