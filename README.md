![Project Screenshot](https://raw.githubusercontent.com/AbhaySinghThakur271/music-genre-classifier/main/Screenshot%20(3).png)
# 🎵 Music Genre Classification

A machine learning system that automatically classifies music tracks into genres — such as **Blues**, **Classical**, **Hip-Hop**, and **Rock** — based on their audio features. This project is an excellent introduction to audio signal processing and multi-class classification.

---

## 📌 Overview

Music genre classification is a classic problem in Music Information Retrieval (MIR). This project extracts meaningful features from raw audio files and trains a machine learning model to predict the genre of a given track.

---

## 🎯 Learning Outcomes

- Audio signal processing and feature extraction (MFCCs, chroma, spectral features)
- Working with non-traditional data types (audio files)
- Building a multi-class classification model
- Applying machine learning to a creative domain

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Librosa, NumPy, Pandas, Scikit-learn, Matplotlib
- **Dataset:** [GTZAN Music Genre Dataset](http://marsyas.info/downloads/datasets.html) (or similar)

---

## 📁 Project Structure

```
music-genre-classification/
│
├── data/                   # Raw and processed audio files
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── preprocess.py       # Feature extraction from audio
│   ├── train.py            # Model training
│   └── predict.py          # Predict genre of a new track
├── models/                 # Saved trained models
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/music-genre-classification.git
cd music-genre-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset

Download the GTZAN dataset (or your own audio dataset) and place it inside the `data/` folder. The folder structure should look like:

```
data/
├── blues/
├── classical/
├── hiphop/
├── rock/
└── ...
```

### 4. Extract Features

```bash
python src/preprocess.py
```

### 5. Train the Model

```bash
python src/train.py
```

### 6. Predict Genre

```bash
python src/predict.py --file path/to/your/audio.wav
```

---

## 📊 Features Extracted

| Feature | Description |
|---|---|
| MFCCs | Mel-Frequency Cepstral Coefficients — captures timbral texture |
| Chroma | Pitch class energy — captures harmony |
| Spectral Centroid | Brightness of the sound |
| Zero Crossing Rate | Rate at which the signal changes sign |
| Spectral Roll-off | Frequency below which 85% of energy is contained |

---

## 🤖 Model

A multi-class classifier (e.g., Random Forest / SVM / Neural Network) is trained on the extracted audio features. The model outputs one of the supported genre labels.

**Supported Genres:**
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## 📈 Results

| Model | Accuracy |
|---|---|
| Random Forest | ~85% |
| SVM | ~83% |
| Neural Network | ~88% |

> Results may vary depending on the dataset split and hyperparameters used.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [ProjectAI](https://projectai.in) for the project inspiration and learning roadmap
- [GTZAN Dataset](http://marsyas.info/downloads/datasets.html) for the audio data
- [Librosa](https://librosa.org/) for audio processing utilities
