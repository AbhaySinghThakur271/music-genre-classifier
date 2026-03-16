# Import necessary libraries
import os
# The json import is no longer needed
import librosa
import numpy as np
import pandas as pd # Ensure pandas is imported

# Define the path to the dataset
DATASET_PATH = "genres_original"

# --- MODIFIED CODE BLOCK ---
# Define the path where the final processed CSV file will be saved
CSV_PATH = "features.csv" 
# The JSON_PATH constant is no longer needed
# --- END MODIFIED CODE BLOCK ---

# Define constants for audio processing
SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10

# Define constants for MFCC extraction
NUM_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Calculate the number of audio samples we expect per 30-second track
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS

# --- MODIFIED FUNCTION SIGNATURE ---
def process_dataset(dataset_path, csv_path):
# --- END MODIFIED FUNCTION SIGNATURE ---
    """
    Extracts features from the dataset and saves them to a CSV file.
    ...
    """
    
    data = {
        "mapping": [],
        "labels": [],

        "features": []
    }

    print("Starting feature extraction...")

    # The main processing loop...
    for i, genre_folder in enumerate(sorted(os.listdir(dataset_path))):
        genre_path = os.path.join(dataset_path, genre_folder)

        if os.path.isdir(genre_path):
            data["mapping"].append(genre_folder)
            print(f"\nProcessing genre: {genre_folder}")

            for filename in sorted(os.listdir(genre_path)):
                if filename.endswith(".wav"):
                    file_path = os.path.join(genre_path, filename)

                    try:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                        
                        if len(signal) >= SAMPLES_PER_TRACK:
                            num_samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)

                            for s in range(NUM_SEGMENTS):
                                start_sample = s * num_samples_per_segment
                                end_sample = start_sample + num_samples_per_segment
                                segment = signal[start_sample:end_sample]

                                # Extract features
                                mfccs_processed = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH), axis=1)
                                chroma_processed = np.mean(librosa.feature.chroma_stft(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH), axis=1)
                                spectral_centroid_processed = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH))
                                spectral_rolloff_processed = np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH))
                                zcr_processed = np.mean(librosa.feature.zero_crossing_rate(y=segment, hop_length=HOP_LENGTH))

                                # Combine all features into a single feature vector
                                feature_vector = np.hstack((mfccs_processed, chroma_processed, spectral_centroid_processed, spectral_rolloff_processed, zcr_processed))
                                
                                # Store the feature vector and its corresponding label
                                data["features"].append(feature_vector.tolist())
                                data["labels"].append(i)

                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
                        continue
    
    # After the loop, convert the data into a pandas DataFrame
    print("\nConverting data to pandas DataFrame...")
    features_df = pd.DataFrame(data["features"])
    features_df["genre_label"] = data["labels"]
    
    # --- THIS IS THE FINAL, NEW CODE BLOCK ---
    # Save the DataFrame to a CSV file
    print(f"Saving DataFrame to {csv_path}...")
    features_df.to_csv(csv_path, index=False)
    # --- END OF THE NEW CODE BLOCK ---
    
    print("\nFeature extraction complete. The file 'features.csv' has been created.")

if __name__ == "__main__":
    # --- MODIFIED FUNCTION CALL ---
    process_dataset(DATASET_PATH, CSV_PATH)
    # --- END MODIFIED FUNCTION CALL ---